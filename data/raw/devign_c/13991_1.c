static abi_long do_sendto(int fd, abi_ulong msg, size_t len, int flags,

                          abi_ulong target_addr, socklen_t addrlen)

{

    void *addr;

    void *host_msg;

    abi_long ret;



    if ((int)addrlen < 0) {

        return -TARGET_EINVAL;

    }



    host_msg = lock_user(VERIFY_READ, msg, len, 1);

    if (!host_msg)

        return -TARGET_EFAULT;

    if (fd_trans_target_to_host_data(fd)) {

        ret = fd_trans_target_to_host_data(fd)(host_msg, len);

        if (ret < 0) {

            unlock_user(host_msg, msg, 0);

            return ret;

        }

    }

    if (target_addr) {

        addr = alloca(addrlen+1);

        ret = target_to_host_sockaddr(fd, addr, target_addr, addrlen);

        if (ret) {

            unlock_user(host_msg, msg, 0);

            return ret;

        }

        ret = get_errno(safe_sendto(fd, host_msg, len, flags, addr, addrlen));

    } else {

        ret = get_errno(safe_sendto(fd, host_msg, len, flags, NULL, 0));

    }

    unlock_user(host_msg, msg, 0);

    return ret;

}
