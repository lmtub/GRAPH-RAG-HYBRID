static VncBasicInfoList *qmp_query_server_entry(QIOChannelSocket *ioc,

                                                bool websocket,

                                                VncBasicInfoList *prev)

{

    VncBasicInfoList *list;

    VncBasicInfo *info;

    Error *err = NULL;

    SocketAddress *addr;



    addr = qio_channel_socket_get_local_address(ioc, &err);

    if (!addr) {

        error_free(err);

        return prev;

    }



    info = g_new0(VncBasicInfo, 1);

    vnc_init_basic_info(addr, info, &err);

    qapi_free_SocketAddress(addr);

    if (err) {

        qapi_free_VncBasicInfo(info);

        error_free(err);

        return prev;

    }

    info->websocket = websocket;



    list = g_new0(VncBasicInfoList, 1);

    list->value = info;

    list->next = prev;

    return list;

}
