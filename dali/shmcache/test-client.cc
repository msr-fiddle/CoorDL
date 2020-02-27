//Test shmem client
// This client assumes that the cache entry exists and
// tries to read content from the cache by attaching to the 
// shm segment

#include <errno.h>
#include <memory>
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <cstdlib>
#include "posixshmem.h"
#include <stdio.h>
#include <sys/mman.h>

using namespace std;


int main (int argc, char* argv[]) {

	int file_size = 0;
	string name = "";

	if (argc != 2){
		cout << "Provide input dir path" << endl;
		exit(1);
	}
	const char* path = argv[1];

	struct dirent *entry = nullptr;
	DIR *dp = nullptr;
	string full_path;
	vector<string> file_paths;
	dp = opendir(path);
	if (dp!= nullptr){
		while ((entry = readdir(dp))){
			if (entry->d_type == DT_REG){
				// Construct full path
				string path_str(path);
				full_path = path_str + "/" + entry->d_name;				
				file_paths.push_back(full_path);
			}
		}
	}
	
	closedir(dp);

	cout << "Read " << file_paths.size() << " files" << endl;
	//for (int i=0; i < file_paths.size(); i++)
	//	cout << i << " : " << file_paths[i] << endl;

	shm::CacheEntry *item = new shm::CacheEntry(file_paths[0]);

	int fd = item->attach_segment();
	if (fd == -1){
		cout << strerror(errno) << endl;
		exit(errno);
	}
	cout << "fd: " << fd << endl;
		
	void* ptr = item->get_cache();
	if (ptr == nullptr){
		cout << strerror(errno) << endl;
		exit(errno);
	}

	int size = item->get_size();

	//just to verify let's write the contents of the pointer to a file
	FILE *f;
	f = fopen("test.JPEG", "wb");
	fwrite(ptr, sizeof(char), size, f);
	fclose(f);

	int ret = 0;

	if ((ret = munmap(ptr, size)) == -1){
		cerr << "Unmap failed" << endl;
		return -1;
	}

	if ((ret = item->close_segment()) < 0){
		cerr << "Close segment failed" << endl;
		exit(errno);
	}

	if ( (ret=item->remove_segment()) < 0 ){
		cerr << "Unlink segment failed" << endl;
		exit(errno);	
	}

	return 0;

}












