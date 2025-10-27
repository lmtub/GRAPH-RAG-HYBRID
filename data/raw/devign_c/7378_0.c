static abi_long do_getpeername(int fd, abi_ulong target_addr,

                               abi_ulong target_addrlen_addr)

{

    socklen_t addrlen;

    void *addr;

    abi_long ret;



    if (get_user_u32(addrlen, target_addrlen_addr))

        return -TARGET_EFAULT;



    if (addrlen < 0 || addrlen > MAX_SOCK_ADDR)

        return -TARGET_EINVAL;



    addr = alloca(addrlen);



    ret = get_errno(getpeername(fd, addr, &addrlen));

    if (!is_error(ret)) {

        host_to_target_sockaddr(target_addr, addr, addrlen);

        if (put_user_u32(addrlen, target_addrlen_addr))

            ret = -TARGET_EFAULT;

    }

    return ret;

}
