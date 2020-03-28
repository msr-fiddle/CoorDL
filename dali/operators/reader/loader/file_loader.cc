// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dirent.h>
#include <errno.h>
#include <memory>
#include <chrono>
#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/operators/shmcache/posixshmem.h"

namespace dali {


//template<int I, int J, class... T>
template<class... T>
inline auto to_pair(std::tuple<T...> t)
    -> decltype(std::make_pair(std::get<0>(t), std::get<1>(t))) {
  return std::make_pair(std::get<0>(t), std::get<1>(t));
}

inline void assemble_file_list(const std::string& path, const std::string& curr_entry, int label,
                        std::vector<std::pair<std::string, int>> *file_label_pairs) {
  std::string curr_dir_path = path + "/" + curr_entry;
  DIR *dir = opendir(curr_dir_path.c_str());

  struct dirent *entry;

  while ((entry = readdir(dir))) {
    std::string full_path = curr_dir_path + "/" + std::string{entry->d_name};
#ifdef _DIRENT_HAVE_D_TYPE
    /*
     * we support only regular files and symlinks, if FS returns DT_UNKNOWN
     * it doesn't mean anything and let us validate filename itself
     */
    if (entry->d_type != DT_REG && entry->d_type != DT_LNK &&
        entry->d_type != DT_UNKNOWN) {
      continue;
    }
#endif
    std::string rel_path = curr_entry + "/" + std::string{entry->d_name};
    if (HasKnownExtension(std::string(entry->d_name))) {
      //file_label_pairs->push_back(std::make_tuple(full_path, label));
      file_label_pairs->push_back(std::make_pair(rel_path, label));
    }
  }
  closedir(dir);
}

inline bool is_cached(std::string name){
   struct stat   buffer;
   const char* name_c = ("/dev/shm/cache/" + name).c_str();
   return (stat (name_c, &buffer) == 0);
}

vector<std::pair<std::string, int>> filesystem::traverse_directories(const std::string& file_root) {
  // open the root
  DIR *dir = opendir(file_root.c_str());

  DALI_ENFORCE(dir != nullptr,
      "Directory " + file_root + " could not be opened.");

  struct dirent *entry;

  std::vector<std::pair<std::string, int>> file_label_pairs;
  std::vector<std::string> entry_name_list;

  while ((entry = readdir(dir))) {
    struct stat s;
    std::string entry_name(entry->d_name);
    std::string full_path = file_root + "/" + entry_name;
    int ret = stat(full_path.c_str(), &s);
    DALI_ENFORCE(ret == 0,
        "Could not access " + full_path + " during directory traversal.");
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
    if (S_ISDIR(s.st_mode)) {
      entry_name_list.push_back(entry_name);
    }
  }
  // sort directories to preserve class alphabetic order, as readdir could
  // return unordered dir list. Otherwise file reader for training and validation
  // could return directories with the same names in completely different order
  std::sort(entry_name_list.begin(), entry_name_list.end());
  for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
      assemble_file_list(file_root, entry_name_list[dir_count], dir_count, &file_label_pairs);
  }
  // sort file names as well
  std::sort(file_label_pairs.begin(), file_label_pairs.end());
  printf("read %lu files from %lu directories\n", file_label_pairs.size(), entry_name_list.size());

  closedir(dir);
  //for (int i = 0; i < static_cast<int>(file_label_pairs.size()); i++)
  //  std::cout << file_label_pairs[i].first << ", " << file_label_pairs[i].second << std::endl;

  return file_label_pairs;
}

void FileLoader::PrepareEmpty(ImageLabelWrapper &image_label) {
  PrepareEmptyTensor(image_label.image);
}

void FileLoader::ReadSample(ImageLabelWrapper &image_label) {
  //std::tuple<std::string, int> image_tuple = image_label_pairs_[current_index_++];
  auto image_pair = image_label_pairs_[current_index_++];
  //auto image_pair = std::make_pair(std::get<0>(image_tuple), std::get<1>(image_tuple));
  int cur_idx = current_index_ - 1;
  //outfile << "Reading Current index = " << cur_idx << ", img = " << image_pair.first << std::endl;

  // handle wrap-around
  MoveToNextShard(current_index_);
  auto start = std::chrono::high_resolution_clock::now();
  // copy the label
  image_label.label = image_pair.second;
  DALIMeta meta;
  meta.SetSourceInfo(image_pair.first);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_pair.first)) {
    meta.SetSkipSample(true);
    image_label.image.Reset();
    image_label.image.SetMeta(meta);
    image_label.image.set_type(TypeInfo::Create<uint8_t>());
    image_label.image.Resize({0});
    return;
  }
  
  /*TODO: All we need to do is update the path
   of the file to point to the shm area instead of disk
   If cache size is not full, then add this entry to cache,
   else do nothing
  */
  //outfile << "SHM cache list length " << shm_cache_index_list_.size() << endl;
  if (cache_size_ > 0){
      bool must_cache = std::binary_search (shm_cache_index_list_.begin(), shm_cache_index_list_.end(), cur_idx);
      //outfile << "Searching for " << image_pair.first << " found : " << must_cache << " cache done? : " << caching_done_ << endl; 
      if (!caching_done_ && must_cache) {
        //outfile << "Must write " << image_pair.first << endl;
        shm::CacheEntry *ce = new shm::CacheEntry(image_pair.first);
        int ret = -1;
        ret = ce->create_segment();
        DALI_ENFORCE(ret != -1,
          "Cache for " + image_pair.first + " could not be created.");

        ret = ce->put_cache_simple(file_root_ + "/" + image_pair.first);
        DALI_ENFORCE(ret != -1,
          "Cache for " + image_pair.first + " could not be populated.");
        //shm_cached_items_.push_back(image_pair.first);

        //outfile << "\twritten : size = " << ce->get_size() << " at " << ce->get_shm_path() << endl; 

        //ret = ce->close_segment();
        //DALI_ENFORCE(ret != -1,
        //  "Cache for " + image_pair.first + " could not be closed.");

        //Update the file path to get a cache hit for its next access.
        //std::get<0>(image_label_pairs_[cur_idx]) = ce->get_shm_path();
        //outfile << "cached " << ce->get_shm_path() << endl;
        delete ce;
      }
  }

  // check if cached
  // Change this to be parameter. Hardcoded for now
  std::string prefix;
  //if (caching_done_ && is_cached(image_pair.first)){
  if (cache_size_ > 0 && caching_done_ && is_cached(image_pair.first)){
    prefix = "/dev/shm/cache";
    //outfile << "Got cached value for " << image_pair.first << endl;
  }
  else
    prefix = file_root_;

  //auto current_image = FileStream::Open(image_pair.first, read_ahead_);
  //auto current_image = FileStream::Open(file_root_ + "/" + image_pair.first, read_ahead_);
  //outfile << "\tReading " << prefix << "/" << image_pair.first << endl;
  auto current_image = FileStream::Open(prefix + "/" + image_pair.first, read_ahead_);
  Index image_size = current_image->Size();

  if (copy_read_data_) {
    //std::cout << "Copying " << image_pair.first << std::endl;
    if (image_label.image.shares_data()) {
      image_label.image.Reset();
    }
    image_label.image.Resize({image_size});
    // copy the image
    current_image->Read(image_label.image.mutable_data<uint8_t>(), image_size);
  } else {
    //std::cout << "Sharing " << image_pair.first << std::endl;
    auto p = current_image->Get(image_size);
    // Wrap the raw data in the Tensor object.
    image_label.image.ShareData(p, image_size, {image_size});
    image_label.image.set_type(TypeInfo::Create<uint8_t>());
  }

  // close the file handle
  current_image->Close();
  auto finish = std::chrono::high_resolution_clock::now();
  //std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

  // copy the label
  image_label.label = image_pair.second;
  image_label.image.SetMeta(meta);
}

Index FileLoader::SizeImpl() {
  return static_cast<Index>(image_label_pairs_.size());
}
}  // namespace dali
