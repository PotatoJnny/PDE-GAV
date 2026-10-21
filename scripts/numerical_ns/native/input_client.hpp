#pragma once
// Local binary protocol, IEEE float64/little endian. Only public u0/f are exposed.
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <stdexcept>
#include <vector>
#include <string>
class NSInputClient {
    int fd_ = -1;
    size_t n_ = 0;
    double deadline_ = 0.;
    void check_deadline() const {
        // Controller charged-time accounting is authoritative. It cleans this
        // inherited process group; socket EOF also stops a disconnected client.

    }
    void sendall(const void* data, size_t size) {
        auto p = static_cast<const char*>(data);
        while(size) { check_deadline(); auto k=::send(fd_,p,size,MSG_NOSIGNAL);
            if(k<0 && (errno==EINTR || errno==EAGAIN || errno==EWOULDBLOCK)) continue;
            if(k<=0) throw std::runtime_error("public-input socket write failed");
            p+=k; size-=k;
        }
    }
    void receive(void* data, size_t size) {
        auto p=static_cast<char*>(data);
        while(size) { check_deadline(); auto k=::recv(fd_,p,size,0);
            if(k<0 && (errno==EINTR || errno==EAGAIN || errno==EWOULDBLOCK)) continue;
            if(k<=0) throw std::runtime_error("public-input socket read failed");
            p+=k; size-=k;
        }
    }
public:
    NSInputClient()=default;
    NSInputClient(const NSInputClient&)=delete;
    NSInputClient& operator=(const NSInputClient&)=delete;
    ~NSInputClient(){ if(fd_>=0) ::close(fd_); }
    void connect(const std::vector<double>& xy) {
        if(xy.empty() || xy.size()%2) throw std::runtime_error("invalid quadrature coordinates");
        check_deadline();
        const char* path=std::getenv("NS_PUBLIC_INPUT_SOCKET");
        sockaddr_un address{};address.sun_family=AF_UNIX;
        if(!path || std::strlen(path)>=sizeof(address.sun_path))
            throw std::runtime_error("missing or oversized public-input socket path");
        std::strcpy(address.sun_path,path);
        fd_=::socket(AF_UNIX,SOCK_STREAM,0);
        if(fd_<0 || ::connect(fd_,reinterpret_cast<sockaddr*>(&address),sizeof(address)))
            throw std::runtime_error("public-input socket connection failed");
        // Poll socket stalls against the SAME original worker deadline. A
        // large valid public callback must not be killed by a fresh 60s cap.
        timeval timeout{1,0};
        ::setsockopt(fd_,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        ::setsockopt(fd_,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        n_=xy.size()/2;uint64_t count=n_;
        const char magic[8]={'N','S','R','E','F','0','1','\n'};
        sendall(magic,8);sendall(&count,8);sendall(xy.data(),xy.size()*8);
        char ack;receive(&ack,1);if(ack!='K') throw std::runtime_error("public-input registration rejected");
    }
    std::vector<double> query(char op,double t) {
        if(op!='I' && op!='F') throw std::runtime_error("non-public operation forbidden");
        sendall(&op,1);sendall(&t,8);
        char ack;receive(&ack,1);if(ack!='K') throw std::runtime_error("public-input evaluation rejected");
        std::vector<double> values(n_*2);receive(values.data(),values.size()*8);return values;
    }
};
