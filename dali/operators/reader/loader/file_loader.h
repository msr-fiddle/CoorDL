// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/util/file.h"

namespace dali {

namespace filesystem {

vector<std::tuple<string, int>> traverse_directories(const std::string& path);

}  // namespace filesystem

struct ImageLabelWrapper {
  Tensor<CPUBackend> image;
  int label;
};

class FileLoader : public Loader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit inline FileLoader(
    const OpSpec& spec,
    vector<std::tuple<string, int>> image_label_pairs = std::vector<std::tuple<string, int>>(),
    bool shuffle_after_epoch = false)
    : Loader<CPUBackend, ImageLabelWrapper>(spec),
      file_root_(spec.GetArgument<string>("file_root")),
      file_list_(spec.GetArgument<string>("file_list")),
      image_label_pairs_(std::move(image_label_pairs)),
      shuffle_after_epoch_(shuffle_after_epoch),
      current_index_(0),
      current_epoch_(0), 
      caching_done_(false) {
      /*
      * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks differently
      * after each epoch so coexistence with `stick_to_shard` doesn't make any sense
      * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLoader so all
      * DALI instances will do shuffling after each epoch
      */
      if (shuffle_after_epoch_ || stick_to_shard_)
        DALI_ENFORCE(
          !shuffle_after_epoch_ || !stick_to_shard_,
          "shuffle_after_epoch and stick_to_shard cannot be both true");
      if (shuffle_after_epoch_ || shuffle_)
        DALI_ENFORCE(
          !shuffle_after_epoch_ || !shuffle_,
          "shuffle_after_epoch and random_shuffle cannot be both true");
      /*
       * Imply `stick_to_shard` from  `shuffle_after_epoch`
       */
      if (shuffle_after_epoch_) {
        stick_to_shard_ = true;
      }
    mmap_reserver = FileStream::FileStreamMappinReserver(
        static_cast<unsigned int>(initial_buffer_fill_));
    copy_read_data_ = !mmap_reserver.CanShareMappedData();
  }

  void PrepareEmpty(ImageLabelWrapper &tensor) override;
  void ReadSample(ImageLabelWrapper &tensor) override;
  
  ~FileLoader() {
     outfile << "Order of shm cached items : " << endl;
     for(int i=0; i < static_cast<int>(shm_cached_items_.size()); i++)
         outfile << i << " : " << shm_cached_items_[i] << endl;
   }

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    std::cout << "Num Shards:" << num_shards_ << "\nShardID:" << shard_id_ << "\nShuffle seed:"<< kDaliDataloaderSeed + shuffle_seed_ << std::endl;
    if (image_label_pairs_.empty()) {
      if (file_list_ == "") {
        image_label_pairs_ = filesystem::traverse_directories(file_root_);
      } else {
        // load (path, label) pairs from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list_);

        string image_file;
        int label;
        while (s >> image_file >> label) {
          auto p = std::make_tuple(image_file, label);
          image_label_pairs_.push_back(p);
        }
        DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list_);
      }
    }
    DALI_ENFORCE(Size() > 0, "No files found.");

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed + shuffle_seed_);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
    Reset(true);
    //for (int i = 0; i < static_cast<int>(image_label_pairs_.size()); i++)
    //    outfile << i << " : " << std::get<0>(image_label_pairs_[i]) << ", " << std::get<1>(image_label_pairs_[i]) << std::endl;
  }

  /* This function is called before the use of FileLoader iter
   * in each epoch. SO any state that needs to be set at 
   * epoch boundaries can be done here
   */
  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, Size());
      index_start_ = current_index_;
      index_end_ = current_index_ + Size()/num_shards_;
      //outfile << __FILE__ << " Reset : wrap to shard" << endl;
    } else {
      current_index_ = 0;
      index_start_ = current_index_;
      index_end_ = current_index_ + Size()/num_shards_;
    }
      outfile << "Current Epoch : " << current_epoch_ << endl;
      outfile << "Cache size : " << cache_size_  << "\nCached : " << caching_done_ << endl;
      outfile << "Current index : " << current_index_ << "\nindex_start : " << index_start_ << "\nindex_end : " << index_end_ << endl;

    // If the epoch count is 1 here, it means we have completed
    // epoch 1. SO stop caching beyond this point
    if (current_epoch_ == 1)
      caching_done_ = true;

    // Create a shuffled list for caching   
    // Sort it so that search becomes easier
    if (!caching_done_ && cache_size_ > 0){
       outfile << "Seed is " << seed_ << endl;
       //shm_cache_index_list_.resize(cache_size_);
       std::mt19937 gen(seed_);
       //std::uniform_int_distribution<int> distr(index_start_, index_end_); 
       //std::generate(shm_cache_index_list_.begin(), shm_cache_index_list_.end(), [&](){ return distr(gen); });
       shm_cache_index_list_.resize(Size()/num_shards_);
       std::iota(shm_cache_index_list_.begin(), shm_cache_index_list_.end(), index_start_);
       std::shuffle(shm_cache_index_list_.begin(), shm_cache_index_list_.end(),gen);
       shm_cache_index_list_.resize(cache_size_);
       std::sort (shm_cache_index_list_.begin(), shm_cache_index_list_.end());

       //outfile << "Index list to cache : " << endl;
       //for (int i = 0; i < static_cast<int>(shm_cache_index_list_.size()); i++)
       //   outfile << i << " : " << shm_cache_index_list_[i] << std::endl;
    }
       

    current_epoch_++;

    if (shuffle_after_epoch_) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
  }

  using Loader<CPUBackend, ImageLabelWrapper>::shard_id_;
  using Loader<CPUBackend, ImageLabelWrapper>::shuffle_seed_;
  using Loader<CPUBackend, ImageLabelWrapper>::cache_size_;
  using Loader<CPUBackend, ImageLabelWrapper>::num_shards_;
  using Loader<CPUBackend, ImageLabelWrapper>::seed_;
  using Loader<CPUBackend, ImageLabelWrapper>::outfile;

  string file_root_, file_list_;

  //A map of file paths, label and a bool that indicates whether cached
  vector<std::tuple<string, int>> image_label_pairs_;
  bool shuffle_after_epoch_;
  Index current_index_;
  int current_epoch_;
  bool caching_done_;
  int index_start_;
  int index_end_;
  vector<int> shm_cache_index_list_;
  vector<std::string> shm_cached_items_;
  FileStream::FileStreamMappinReserver mmap_reserver;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
