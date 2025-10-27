static abi_long do_setsockopt(int sockfd, int level, int optname,

                              abi_ulong optval_addr, socklen_t optlen)

{

    abi_long ret;

    int val;

    struct ip_mreqn *ip_mreq;

    struct ip_mreq_source *ip_mreq_source;



    switch(level) {

    case SOL_TCP:

        /* TCP options all take an 'int' value.  */

        if (optlen < sizeof(uint32_t))

            return -TARGET_EINVAL;



        if (get_user_u32(val, optval_addr))

            return -TARGET_EFAULT;

        ret = get_errno(setsockopt(sockfd, level, optname, &val, sizeof(val)));

        break;

    case SOL_IP:

        switch(optname) {

        case IP_TOS:

        case IP_TTL:

        case IP_HDRINCL:

        case IP_ROUTER_ALERT:

        case IP_RECVOPTS:

        case IP_RETOPTS:

        case IP_PKTINFO:

        case IP_MTU_DISCOVER:

        case IP_RECVERR:

        case IP_RECVTOS:

#ifdef IP_FREEBIND

        case IP_FREEBIND:

#endif

        case IP_MULTICAST_TTL:

        case IP_MULTICAST_LOOP:

            val = 0;

            if (optlen >= sizeof(uint32_t)) {

                if (get_user_u32(val, optval_addr))

                    return -TARGET_EFAULT;

            } else if (optlen >= 1) {

                if (get_user_u8(val, optval_addr))

                    return -TARGET_EFAULT;

            }

            ret = get_errno(setsockopt(sockfd, level, optname, &val, sizeof(val)));

            break;

        case IP_ADD_MEMBERSHIP:

        case IP_DROP_MEMBERSHIP:

            if (optlen < sizeof (struct target_ip_mreq) ||

                optlen > sizeof (struct target_ip_mreqn))

                return -TARGET_EINVAL;



            ip_mreq = (struct ip_mreqn *) alloca(optlen);

            target_to_host_ip_mreq(ip_mreq, optval_addr, optlen);

            ret = get_errno(setsockopt(sockfd, level, optname, ip_mreq, optlen));

            break;



        case IP_BLOCK_SOURCE:

        case IP_UNBLOCK_SOURCE:

        case IP_ADD_SOURCE_MEMBERSHIP:

        case IP_DROP_SOURCE_MEMBERSHIP:

            if (optlen != sizeof (struct target_ip_mreq_source))

                return -TARGET_EINVAL;



            ip_mreq_source = lock_user(VERIFY_READ, optval_addr, optlen, 1);

            ret = get_errno(setsockopt(sockfd, level, optname, ip_mreq_source, optlen));

            unlock_user (ip_mreq_source, optval_addr, 0);

            break;



        default:

            goto unimplemented;

        }

        break;

    case TARGET_SOL_SOCKET:

        switch (optname) {

            /* Options with 'int' argument.  */

        case TARGET_SO_DEBUG:

		optname = SO_DEBUG;

		break;

        case TARGET_SO_REUSEADDR:

		optname = SO_REUSEADDR;

		break;

        case TARGET_SO_TYPE:

		optname = SO_TYPE;

		break;

        case TARGET_SO_ERROR:

		optname = SO_ERROR;

		break;

        case TARGET_SO_DONTROUTE:

		optname = SO_DONTROUTE;

		break;

        case TARGET_SO_BROADCAST:

		optname = SO_BROADCAST;

		break;

        case TARGET_SO_SNDBUF:

		optname = SO_SNDBUF;

		break;

        case TARGET_SO_RCVBUF:

		optname = SO_RCVBUF;

		break;

        case TARGET_SO_KEEPALIVE:

		optname = SO_KEEPALIVE;

		break;

        case TARGET_SO_OOBINLINE:

		optname = SO_OOBINLINE;

		break;

        case TARGET_SO_NO_CHECK:

		optname = SO_NO_CHECK;

		break;

        case TARGET_SO_PRIORITY:

		optname = SO_PRIORITY;

		break;

#ifdef SO_BSDCOMPAT

        case TARGET_SO_BSDCOMPAT:

		optname = SO_BSDCOMPAT;

		break;

#endif

        case TARGET_SO_PASSCRED:

		optname = SO_PASSCRED;

		break;

        case TARGET_SO_TIMESTAMP:

		optname = SO_TIMESTAMP;

		break;

        case TARGET_SO_RCVLOWAT:

		optname = SO_RCVLOWAT;

		break;

        case TARGET_SO_RCVTIMEO:

		optname = SO_RCVTIMEO;

		break;

        case TARGET_SO_SNDTIMEO:

		optname = SO_SNDTIMEO;

		break;

            break;

        default:

            goto unimplemented;

        }

	if (optlen < sizeof(uint32_t))

            return -TARGET_EINVAL;



	if (get_user_u32(val, optval_addr))

            return -TARGET_EFAULT;

	ret = get_errno(setsockopt(sockfd, SOL_SOCKET, optname, &val, sizeof(val)));

        break;

    default:

    unimplemented:

        gemu_log("Unsupported setsockopt level=%d optname=%d \n", level, optname);

        ret = -TARGET_ENOPROTOOPT;

    }

    return ret;

}
