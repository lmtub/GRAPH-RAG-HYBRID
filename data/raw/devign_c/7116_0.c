static void vnc_init_basic_info_from_remote_addr(QIOChannelSocket *ioc,

                                                 VncBasicInfo *info,

                                                 Error **errp)

{

    SocketAddress *addr = NULL;



    addr = qio_channel_socket_get_remote_address(ioc, errp);

    if (!addr) {

        return;

    }



    vnc_init_basic_info(addr, info, errp);

    qapi_free_SocketAddress(addr);

}
