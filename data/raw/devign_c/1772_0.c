static abi_long do_socketcall(int num, abi_ulong vptr)

{

    abi_long ret;

    const int n = sizeof(abi_ulong);



    switch(num) {

    case SOCKOP_socket:

	{

            abi_ulong domain, type, protocol;



            if (get_user_ual(domain, vptr)

                || get_user_ual(type, vptr + n)

                || get_user_ual(protocol, vptr + 2 * n))

                return -TARGET_EFAULT;



            ret = do_socket(domain, type, protocol);

	}

        break;

    case SOCKOP_bind:

	{

            abi_ulong sockfd;

            abi_ulong target_addr;

            socklen_t addrlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(target_addr, vptr + n)

                || get_user_ual(addrlen, vptr + 2 * n))

                return -TARGET_EFAULT;



            ret = do_bind(sockfd, target_addr, addrlen);

        }

        break;

    case SOCKOP_connect:

        {

            abi_ulong sockfd;

            abi_ulong target_addr;

            socklen_t addrlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(target_addr, vptr + n)

                || get_user_ual(addrlen, vptr + 2 * n))

                return -TARGET_EFAULT;



            ret = do_connect(sockfd, target_addr, addrlen);

        }

        break;

    case SOCKOP_listen:

        {

            abi_ulong sockfd, backlog;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(backlog, vptr + n))

                return -TARGET_EFAULT;



            ret = get_errno(listen(sockfd, backlog));

        }

        break;

    case SOCKOP_accept:

        {

            abi_ulong sockfd;

            abi_ulong target_addr, target_addrlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(target_addr, vptr + n)

                || get_user_ual(target_addrlen, vptr + 2 * n))

                return -TARGET_EFAULT;



            ret = do_accept4(sockfd, target_addr, target_addrlen, 0);

        }

        break;

    case SOCKOP_accept4:

        {

            abi_ulong sockfd;

            abi_ulong target_addr, target_addrlen;

            abi_ulong flags;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(target_addr, vptr + n)

                || get_user_ual(target_addrlen, vptr + 2 * n)

                || get_user_ual(flags, vptr + 3 * n)) {

                return -TARGET_EFAULT;

            }



            ret = do_accept4(sockfd, target_addr, target_addrlen, flags);

        }

        break;

    case SOCKOP_getsockname:

        {

            abi_ulong sockfd;

            abi_ulong target_addr, target_addrlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(target_addr, vptr + n)

                || get_user_ual(target_addrlen, vptr + 2 * n))

                return -TARGET_EFAULT;



            ret = do_getsockname(sockfd, target_addr, target_addrlen);

        }

        break;

    case SOCKOP_getpeername:

        {

            abi_ulong sockfd;

            abi_ulong target_addr, target_addrlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(target_addr, vptr + n)

                || get_user_ual(target_addrlen, vptr + 2 * n))

                return -TARGET_EFAULT;



            ret = do_getpeername(sockfd, target_addr, target_addrlen);

        }

        break;

    case SOCKOP_socketpair:

        {

            abi_ulong domain, type, protocol;

            abi_ulong tab;



            if (get_user_ual(domain, vptr)

                || get_user_ual(type, vptr + n)

                || get_user_ual(protocol, vptr + 2 * n)

                || get_user_ual(tab, vptr + 3 * n))

                return -TARGET_EFAULT;



            ret = do_socketpair(domain, type, protocol, tab);

        }

        break;

    case SOCKOP_send:

        {

            abi_ulong sockfd;

            abi_ulong msg;

            size_t len;

            abi_ulong flags;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(msg, vptr + n)

                || get_user_ual(len, vptr + 2 * n)

                || get_user_ual(flags, vptr + 3 * n))

                return -TARGET_EFAULT;



            ret = do_sendto(sockfd, msg, len, flags, 0, 0);

        }

        break;

    case SOCKOP_recv:

        {

            abi_ulong sockfd;

            abi_ulong msg;

            size_t len;

            abi_ulong flags;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(msg, vptr + n)

                || get_user_ual(len, vptr + 2 * n)

                || get_user_ual(flags, vptr + 3 * n))

                return -TARGET_EFAULT;



            ret = do_recvfrom(sockfd, msg, len, flags, 0, 0);

        }

        break;

    case SOCKOP_sendto:

        {

            abi_ulong sockfd;

            abi_ulong msg;

            size_t len;

            abi_ulong flags;

            abi_ulong addr;

            abi_ulong addrlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(msg, vptr + n)

                || get_user_ual(len, vptr + 2 * n)

                || get_user_ual(flags, vptr + 3 * n)

                || get_user_ual(addr, vptr + 4 * n)

                || get_user_ual(addrlen, vptr + 5 * n))

                return -TARGET_EFAULT;



            ret = do_sendto(sockfd, msg, len, flags, addr, addrlen);

        }

        break;

    case SOCKOP_recvfrom:

        {

            abi_ulong sockfd;

            abi_ulong msg;

            size_t len;

            abi_ulong flags;

            abi_ulong addr;

            socklen_t addrlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(msg, vptr + n)

                || get_user_ual(len, vptr + 2 * n)

                || get_user_ual(flags, vptr + 3 * n)

                || get_user_ual(addr, vptr + 4 * n)

                || get_user_ual(addrlen, vptr + 5 * n))

                return -TARGET_EFAULT;



            ret = do_recvfrom(sockfd, msg, len, flags, addr, addrlen);

        }

        break;

    case SOCKOP_shutdown:

        {

            abi_ulong sockfd, how;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(how, vptr + n))

                return -TARGET_EFAULT;



            ret = get_errno(shutdown(sockfd, how));

        }

        break;

    case SOCKOP_sendmsg:

    case SOCKOP_recvmsg:

        {

            abi_ulong fd;

            abi_ulong target_msg;

            abi_ulong flags;



            if (get_user_ual(fd, vptr)

                || get_user_ual(target_msg, vptr + n)

                || get_user_ual(flags, vptr + 2 * n))

                return -TARGET_EFAULT;



            ret = do_sendrecvmsg(fd, target_msg, flags,

                                 (num == SOCKOP_sendmsg));

        }

        break;

    case SOCKOP_setsockopt:

        {

            abi_ulong sockfd;

            abi_ulong level;

            abi_ulong optname;

            abi_ulong optval;

            abi_ulong optlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(level, vptr + n)

                || get_user_ual(optname, vptr + 2 * n)

                || get_user_ual(optval, vptr + 3 * n)

                || get_user_ual(optlen, vptr + 4 * n))

                return -TARGET_EFAULT;



            ret = do_setsockopt(sockfd, level, optname, optval, optlen);

        }

        break;

    case SOCKOP_getsockopt:

        {

            abi_ulong sockfd;

            abi_ulong level;

            abi_ulong optname;

            abi_ulong optval;

            socklen_t optlen;



            if (get_user_ual(sockfd, vptr)

                || get_user_ual(level, vptr + n)

                || get_user_ual(optname, vptr + 2 * n)

                || get_user_ual(optval, vptr + 3 * n)

                || get_user_ual(optlen, vptr + 4 * n))

                return -TARGET_EFAULT;



            ret = do_getsockopt(sockfd, level, optname, optval, optlen);

        }

        break;

    default:

        gemu_log("Unsupported socketcall: %d\n", num);

        ret = -TARGET_ENOSYS;

        break;

    }

    return ret;

}
