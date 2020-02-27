//Test shmem

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

using namespace std;


/*
struct recursive_directory_range
{
    typedef recursive_directory_iterator iterator;
    recursive_directory_range(path p) : p_(p) {}

    iterator begin() { return recursive_directory_iterator(p_); }
    iterator end() { return recursive_directory_iterator(); }

    path p_;
};
*/

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
	//for (auto it : recursive_directory_range(dir_path))
	//	file_paths.push_back(it);

	cout << "Read " << file_paths.size() << " files" << endl;
	//for (int i=0; i < file_paths.size(); i++)
	//	cout << i << " : " << file_paths[i] << endl;

	shm::CacheEntry *item = new shm::CacheEntry(file_paths[0]);
	int fd = item->create_segment();
	if (fd == -1){
		cout << strerror(errno) << endl;
		exit(errno);
	}

	int fd1 = item->attach_segment();
	if (fd1 == -1){
		cout << strerror(errno) << endl;
		exit(errno);
	}
	cout << "fd1: " << fd << ", fd2: " << fd1 << endl;
		
	int written = item->put_cache(file_paths[0]);
	if (written < 0){
		cout << strerror(errno) << endl;
		exit(errno);
	}

	return 0;


}












