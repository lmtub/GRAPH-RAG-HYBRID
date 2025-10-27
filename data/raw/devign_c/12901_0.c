static abi_long do_sendrecvmsg(int fd, abi_ulong target_msg,

                               int flags, int send)

{

    abi_long ret, len;

    struct target_msghdr *msgp;

    struct msghdr msg;

    int count;

    struct iovec *vec;

    abi_ulong target_vec;



    /* FIXME */

    if (!lock_user_struct(send ? VERIFY_READ : VERIFY_WRITE,

                          msgp,

                          target_msg,

                          send ? 1 : 0))

        return -TARGET_EFAULT;

    if (msgp->msg_name) {

        msg.msg_namelen = tswap32(msgp->msg_namelen);

        msg.msg_name = alloca(msg.msg_namelen);

        ret = target_to_host_sockaddr(msg.msg_name, tswapal(msgp->msg_name),

                                msg.msg_namelen);

        if (ret) {

            unlock_user_struct(msgp, target_msg, send ? 0 : 1);

            return ret;

        }

    } else {

        msg.msg_name = NULL;

        msg.msg_namelen = 0;

    }

    msg.msg_controllen = 2 * tswapal(msgp->msg_controllen);

    msg.msg_control = alloca(msg.msg_controllen);

    msg.msg_flags = tswap32(msgp->msg_flags);



    count = tswapal(msgp->msg_iovlen);

    vec = alloca(count * sizeof(struct iovec));

    target_vec = tswapal(msgp->msg_iov);

    lock_iovec(send ? VERIFY_READ : VERIFY_WRITE, vec, target_vec, count, send);

    msg.msg_iovlen = count;

    msg.msg_iov = vec;



    if (send) {

        ret = target_to_host_cmsg(&msg, msgp);

        if (ret == 0)

            ret = get_errno(sendmsg(fd, &msg, flags));

    } else {

        ret = get_errno(recvmsg(fd, &msg, flags));

        if (!is_error(ret)) {

            len = ret;

            ret = host_to_target_cmsg(msgp, &msg);

            if (!is_error(ret)) {

                msgp->msg_namelen = tswap32(msg.msg_namelen);

                if (msg.msg_name != NULL) {

                    ret = host_to_target_sockaddr(tswapal(msgp->msg_name),

                                    msg.msg_name, msg.msg_namelen);

                    if (ret) {

                        goto out;

                    }

                }



                ret = len;

            }

        }

    }



out:

    unlock_iovec(vec, target_vec, count, !send);

    unlock_user_struct(msgp, target_msg, send ? 0 : 1);

    return ret;

}
