static void test_io_channel_setup_sync(SocketAddressLegacy *listen_addr,

                                       SocketAddressLegacy *connect_addr,

                                       QIOChannel **src,

                                       QIOChannel **dst)

{

    QIOChannelSocket *lioc;



    lioc = qio_channel_socket_new();

    qio_channel_socket_listen_sync(lioc, listen_addr, &error_abort);



    if (listen_addr->type == SOCKET_ADDRESS_LEGACY_KIND_INET) {

        SocketAddressLegacy *laddr = qio_channel_socket_get_local_address(

            lioc, &error_abort);



        g_free(connect_addr->u.inet.data->port);

        connect_addr->u.inet.data->port = g_strdup(laddr->u.inet.data->port);



        qapi_free_SocketAddressLegacy(laddr);

    }



    *src = QIO_CHANNEL(qio_channel_socket_new());

    qio_channel_socket_connect_sync(

        QIO_CHANNEL_SOCKET(*src), connect_addr, &error_abort);

    qio_channel_set_delay(*src, false);



    qio_channel_wait(QIO_CHANNEL(lioc), G_IO_IN);

    *dst = QIO_CHANNEL(qio_channel_socket_accept(lioc, &error_abort));

    g_assert(*dst);



    test_io_channel_set_socket_bufs(*src, *dst);



    object_unref(OBJECT(lioc));

}
