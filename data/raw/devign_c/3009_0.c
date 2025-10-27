static abi_long do_connect(int sockfd, abi_ulong target_addr,

                           socklen_t addrlen)

{

    void *addr;



    if (addrlen < 0)

        return -TARGET_EINVAL;



    addr = alloca(addrlen);



    target_to_host_sockaddr(addr, target_addr, addrlen);

    return get_errno(connect(sockfd, addr, addrlen));

}
