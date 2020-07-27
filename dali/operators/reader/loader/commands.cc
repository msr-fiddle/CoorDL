#include "commands.h"

namespace dali {

void print_socket_options (int sockfd) {
    socklen_t i;
    size_t len;
    //size_t t1, t2;

    i = sizeof(len);
    if (getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &len, &i) < 0) {
        std::cerr << "Error getting recvbuf size" << strerror(errno) << std::endl;
        return;
    }
    std::cout << "Receive Buf size for " << sockfd << " : " << len << std::endl;


    if (getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &len, &i) < 0) {
        std::cerr << "Error getting sendbuf size" << strerror(errno) << std::endl;
        return;
    }
    std::cout << "Send Buf size for " << sockfd << " : " << len << std::endl;


    if (getsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &len, &i) < 0) {
        std::cerr << "Error getting TCP nodelay size" << strerror(errno) << std::endl;
        return;
    }
    std::cout << "TCP nodelay for " << sockfd << " : " << len << std::endl;


    if (getsockopt(sockfd, IPPROTO_TCP, TCP_MAXSEG, &len, &i) < 0) {
        std::cerr << "Error getting TCP MSS size" << strerror(errno) << std::endl;
        return;
    }
    std::cout << "TCP MSS for " << sockfd << " : " << len << std::endl;
}


bool set_recv_window(int sockfd, int len_bytes) { 
    socklen_t i;
    i = sizeof(int);

    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &len_bytes, i) < 0) {
        std::cerr << "Error setting recvbuf size" << strerror(errno) << std::endl;
        return false;
    }
    return true;
}


bool set_send_window(int sockfd, int len_bytes) { 
    socklen_t i;
    i = sizeof(int);

    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &len_bytes, i) < 0) {
        std::cerr << "Error setting sendbuf size" << strerror(errno) << std::endl;
        return false;
    }
    return true;
}


bool set_tcp_nodelay(int sockfd) { 
    int yes = 1;

    if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char*) &yes, sizeof(int)) < 0) {
        std::cerr << "Error setting tcp nodel" << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

} //namespace dali
