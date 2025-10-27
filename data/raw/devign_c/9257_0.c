static abi_long do_recvfrom(int fd, abi_ulong msg, size_t len, int flags,

                            abi_ulong target_addr,

                            abi_ulong target_addrlen)

{

    socklen_t addrlen;

    void *addr;

    void *host_msg;

    abi_long ret;



    host_msg = lock_user(VERIFY_WRITE, msg, len, 0);

    if (!host_msg)

        return -TARGET_EFAULT;

    if (target_addr) {

        if (get_user_u32(addrlen, target_addrlen)) {

            ret = -TARGET_EFAULT;

            goto fail;

        }

        if (addrlen < 0 || addrlen > MAX_SOCK_ADDR) {

            ret = -TARGET_EINVAL;

            goto fail;

        }

        addr = alloca(addrlen);

        ret = get_errno(recvfrom(fd, host_msg, len, flags, addr, &addrlen));

    } else {

        addr = NULL; /* To keep compiler quiet.  */

        ret = get_errno(recv(fd, host_msg, len, flags));

    }

    if (!is_error(ret)) {

        if (target_addr) {

            host_to_target_sockaddr(target_addr, addr, addrlen);

            if (put_user_u32(addrlen, target_addrlen)) {

                ret = -TARGET_EFAULT;

                goto fail;

            }

        }

        unlock_user(host_msg, msg, len);

    } else {

fail:

        unlock_user(host_msg, msg, 0);

    }

    return ret;

}
