// A utility file to create, attach, read and write to shared memory segments


#include <errno.h>
#include <memory>
#include <iostream>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

#include "posixshmem.h"
using namespace std;

namespace shm {

string create_name(string path) {
	std::replace( path.begin(), path.end(), '/', '_');	
	return path;
}

string shm_path(string name, string prefix){
	// assumes prefix ends with '/'
	return prefix + name;
}

int open_shared_file(const char* path, int flags, mode_t mode) {
	if (!path){
		errno = ENOENT;
	}
	
	flags |= O_NOFOLLOW | O_CLOEXEC;
	/* Disable asynchronous cancellation.  */
	int state;
	pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &state);
	int fd = open (path, flags, mode);
	if (fd == -1){
		cerr << "Cannot open shm segment" << endl;
	}
	pthread_setcancelstate (state, NULL);
	return fd;
}

int get_file_size(string filename){
	struct stat st;
	stat(filename.c_str(), &st);
	int size = st.st_size;
	return size;
}

CacheEntry::CacheEntry(string path){
	/* If tyhe path conatins prefix, 
	 * (can happen when we are attching 
	 * to an existing cache segment),  
	 * then the name of the segment is 
	 * path - prefix 
	 */
	path_ = path; 
	int found = path.find(prefix); 
	if(found != string::npos){ 
		// Prefix is found in path
		path_.erase(found, prefix.length()); 
	}
	name_ = path_;
	std::replace( name_.begin(), name_.end(), '/', '_');
}

int CacheEntry::create_segment() {
	// Get the unique name for the shm segment
	//name_ =  create_name(path_);
	int flags = O_CREAT | O_RDWR;
	int mode = 511;
	//Get the full shm path and open it
	string shm_path_name = shm_path(name_, prefix);
	fd_ = open_shared_file(shm_path_name.c_str(), flags, mode);
	return fd_;
}

int CacheEntry::attach_segment(){
	// if the shm segment is already open,
	// return the descriptor
	if (fd_ != -1)
		return fd_;

	// Else, open the file without the O_CREAT
	// flags and return the fd
	int flags = O_RDWR;
	int mode = 511;
	string shm_path_name;

	shm_path_name = shm_path(name_, prefix);

	fd_ = open_shared_file(shm_path_name.c_str(), flags, mode);	
	return fd_;
}



int CacheEntry::put_cache(string from_file) {
	int bytes_to_write = get_file_size(from_file);
	size_ = bytes_to_write;
	cout << "will write from file " << from_file << " size " << bytes_to_write << endl;
	if (fd_ < 0){
		errno = EINVAL;
		cerr << "File " << name_ << " has invalid decriptor" << endl;
		return -1;
	}
	ftruncate(fd_, bytes_to_write);

	//mmap the shm file to get ptr
	void *ptr = nullptr;
	if ((ptr = mmap(0, bytes_to_write, PROT_WRITE, MAP_SHARED, fd_, 0)) == MAP_FAILED){
		cerr << "mmap error" << endl;  
		return -1;
	} 
	

	// write to shared memory segment
	// We will mmap the file to read from, because
	// in DALI, the file to be read will be mmaped first. 
	void *ptr_from = nullptr;
	int fd_from = -1;
	if ((fd_from = open(from_file.c_str(), O_RDONLY)) < 0) { 
		cerr << "Open failed" << endl; 
		return -1;
	}
	if ((ptr_from = mmap(0, bytes_to_write, PROT_READ, MAP_SHARED, fd_from, 0)) == MAP_FAILED){
		cerr << "mmap error" << endl; 
		return -1;
	}
	std::shared_ptr<void> p_;
	p_ = shared_ptr<void>(ptr_from, [=](void*) {
		munmap(ptr_from, bytes_to_write); 
	});
	
	//Do the memcpy now
	// memcpy(void* dest, const void* src, size)
	memcpy(ptr, p_.get(), bytes_to_write);
	cout << "memcpy done" << endl;
	int ret = 0;

	// Now unmap both files
	if ((ret = munmap(ptr, bytes_to_write)) == -1){
		cerr << "Munmap failed" << endl;
		return -1;
	}
	
	if ((ret = munmap(ptr_from, bytes_to_write)) == -1){
		cerr << "Munmap failed" << endl;
		return -1;
	}

	close(fd_from);

	return bytes_to_write;
}


void* CacheEntry::get_cache() {
	string from_file = prefix + name_;
	int bytes_to_read = get_file_size(from_file);
	size_ = bytes_to_read;
	cout << "will read from file " << from_file << " size " << bytes_to_read << endl;

	// If the descriptor is invalid, you need to sttach the segment.
	if (fd_ < 0){
		errno = EINVAL;
		cerr << "File " << name_ << " has invalid decriptor" << endl;
		return nullptr;
	}
	//mmap the shm file to get ptr
	void *ptr = nullptr;   
	if ((ptr = mmap(0, bytes_to_read, PROT_READ, MAP_SHARED, fd_, 0)) == MAP_FAILED){
		cerr << "mmap error" << endl;
		return nullptr;
	}

	return ptr;
}

string CacheEntry::get_shm_path(){
	string shm_path_name; 
	shm_path_name = shm_path(name_, prefix);
	return shm_path_name;
}

int CacheEntry::close_segment(){
	int ret = 0;
	if (fd_ > -1){
		if (( ret = close(fd_)) < 0){
			cerr << "File " << prefix + name_ << " close failed" << endl;
			return -1;
		}
	}
	return 0;
}

int CacheEntry::remove_segment(){
	string shm_path_name;
	shm_path_name = shm_path(name_, prefix); 
	int result = unlink(shm_path_name.c_str());
	return result;
}

} //end namespace shm
