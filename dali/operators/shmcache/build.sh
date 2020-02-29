g++ test-server.cc posixshmem.cc posixshmem.h -std=gnu++0x -lrt -o server
g++ test-client.cc posixshmem.cc posixshmem.h -std=gnu++0x -lrt -o client
#g++ test-server.cc -lrt --std=c++17 -lstdc++fs -Wall -o server
