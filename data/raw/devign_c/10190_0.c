static abi_long do_socket(int domain, int type, int protocol)

{

    int target_type = type;

    int ret;



    ret = target_to_host_sock_type(&type);

    if (ret) {

        return ret;

    }



    if (domain == PF_NETLINK)

        return -TARGET_EAFNOSUPPORT;



    if (domain == AF_PACKET ||

        (domain == AF_INET && type == SOCK_PACKET)) {

        protocol = tswap16(protocol);

    }



    ret = get_errno(socket(domain, type, protocol));

    if (ret >= 0) {

        ret = sock_flags_fixup(ret, target_type);

        if (type == SOCK_PACKET) {

            /* Manage an obsolete case :

             * if socket type is SOCK_PACKET, bind by name

             */

            fd_trans_register(ret, &target_packet_trans);

        }

    }

    return ret;

}
