// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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


#include "dali/operators/reader/loader/loader.h"

namespace dali {

DALI_SCHEMA(LoaderBase)
  .AddOptionalArg("random_shuffle",
      R"code(Whether to randomly shuffle data. Prefetch buffer of `initial_fill` size is used
to sequentially read data and then randomly sample it to form a batch.)code", false)
  .AddOptionalArg("initial_fill",
      R"code(Size of the buffer used for shuffling. If `random_shuffle` is off then
this parameter is ignored.)code", 1024)
  .AddOptionalArg("num_shards",
      R"code(Partition the data into this many parts (used for multiGPU training).)code", 1)
  .AddOptionalArg("num_nodes",
      R"code(Number of physical nodes involved in multi GPU training).)code", 1)
  .AddOptionalArg("node_id",
      R"code(ID of the node in multi GPU training).)code", 0)
  .AddOptionalArg("shard_id",
      R"code(Id of the part to read.)code", 0)
  .AddOptionalArg("cache_size",
      R"code(Number of items to cache for this loader.)code", 0)
  .AddOptionalArg("shuffle_seed",
      R"code(Shuffle the initial dataset metadata)code", 0)
  .AddOptionalArg("tensor_init_bytes",
      R"code(Hint for how much memory to allocate per image.)code", 1048576)
  .AddOptionalArg("stick_to_shard",
      R"code(Whether reader should stick to given data shard instead of going through the whole dataset.
When decoder caching is used, it reduces significantly the amount of data to be cached, but could affect
accuracy in some cases)code", false)
  .AddOptionalArg("read_ahead",
      R"code(Whether accessed data should be read ahead. In case of big files like LMDB,
RecordIO or TFRecord it will slow down first access but will decrease the time of all following
accesses.)code", false)
  .AddOptionalArg("prefetch_queue_depth",
      R"code(Specifies the number of batches prefetched by the internal Loader. To be increased when pipeline
processing is CPU stage-bound, trading memory consumption for better interleaving with the Loader thread.)code", 1)
  .AddOptionalArg("skip_cached_images",
      R"code(If set to true, loading data will be skipped when the sample is present in the decoder cache.
In such case the output of the loader will be empty)code", false)
  .AddOptionalArg("lazy_init",
      R"code(If set to true, Loader will parse and prepare the dataset metadata only during the first `Run`
instead of in the constructor.)code", false)
  .AddOptionalArg("pad_last_batch",
      R"code(If set to true, the Loader will pad the last batch with the last image when the batch size is
not aligned with the shard size. It means that the remainder of the batch or even the whole batch can be
artificially added when the data set size is not equally divisible by the number of shards, and the shard is
not equally divisible by the batch size. In the end, the shard size will be equalized between shards.)code", false);

size_t start_index(const size_t shard_id,
                   const size_t shard_num,
                   const size_t size) {
  return size * shard_id / shard_num;
}


size_t start_index_for_node(const size_t node_id,
                   const size_t num_nodes,
                   const size_t size) {
  return size * node_id / num_nodes;
}


Index num_samples(const size_t shard_num,
                  const size_t size) {
  return static_cast<Index>(std::ceil(size * 1.0 / shard_num));
}

int initialize_socket(int port, string ip_addr) {
  int cnct, server_fd;
  struct sockaddr_in server;
  struct hostent *hp;

  server.sin_family=AF_INET;
  server.sin_port=htons(port);
  server.sin_addr.s_addr=INADDR_ANY;
  server_fd =socket(AF_INET,SOCK_STREAM,0);
  if(     server_fd       <       0){
    cout<<"Error creating socket\n";
    return -1;
  }

  cout<< " Socket created for " << port << endl;
  hp=gethostbyname(ip_addr.c_str());
  bcopy((char *)hp->h_addr,(char *)&server.sin_addr.s_addr,hp->h_length);
  cnct=connect(server_fd,(struct sockaddr*)&server,sizeof(server));
  if(cnct<0){    
    cout<<"Error connecting " << endl;    
    return -1;  
  }

  int yes = 1;
  int result = setsockopt(server_fd, 
													IPPROTO_TCP,
													TCP_NODELAY,
													(char *) &yes, 
													sizeof(int));  
	if (result < 0){
    cout<<"Error setting NO_DELAY " << endl;    
    return -1;  
  }
  cout<<"Connection has been made for " << ip_addr  << " : " << port << endl;
    return server_fd;
}


}  // namespace dali
