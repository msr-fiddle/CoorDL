#ifndef POSIX_SHMEM_H_
#define POSIX_SHMEM_H_

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


namespace shm{

extern const std::string prefix;  

class CacheEntry {
    public:

	/* Constructor : initialize segment using path of
	 * disk file, and populate name. This does not 
	 * open the file until create or attach segment 
	 * is called
	 */
	CacheEntry(std::string path);


	/* Given a path to the current file on disk 
	 * that has to be cached in shared memory
	 * this function will return the descriptor
	 * for the corresponding shared memory 
	 * segment. If the segment already exists,
	 * it returns the fd. If not, it creates and then
	 * returns the handle
	 */
        int create_segment();

	/* This function must be used when you
	 * know the shm segment exists and you want a
	 * handle to it. If the segment does not exist,
	 * the function returns EINVAL error
	 */
	int attach_segment();
	
	/* Given the path to the shm segment or a name
	 * this function returns the pointer to the
	 * contents of the file after mmap in its
	 * address space. We don't implement this
	 * because DALI already does this
	 * It mmaps a path, and reads it. We'll update
	 * the path to file -> shm
	 * In the tmp test implementation, always 
	 * unmap after reading
	 */
	void* get_cache();

	/* Write to the shm segment that is already 
	 * created and open by mmap into its address space
	 * unmaps on exit.
	 */
	int put_cache(std::string from_file);
	int put_cache_simple(std::string from_file);

	/* This function returns the shared memory path
	 * to this segment. This path can be 
	 * updated in the dali file map.
	 */
	std::string get_shm_path();

	/* Returns the file size
	*/
	int get_size() { return size_;}

	/* This function must be called after
	 * an attach_segment or create_segment
	 * It closes the opened segment, but 
	 * does not delete it.
	 */
	int close_segment();

	/* Thisfunction must be called to delete 
	 * the cache.
	 * Ideally, this must be called at the end 
	 * of experiment when you want to c;lear off
	 * all entries
	 */
	int remove_segment();

	/* name_ is the identifier of the shared
	 * segment in the path indicated by prefix
	 * To access the segment use path :
	 * prefix + name
	 */
	std::string name_;

	/* path is the path of the source file
	 * whose content will be stored in the
	 * shared segment
	 */
	//std::string path_;

    private:
        //fd of the open file - tmp if create-segment
        // final if attach-segment
	int fd_ = -1;

	int size_ = 0;
	//std::string prefix = "/dev/shm/cache/";
};

} // end namespace shm

#endif
