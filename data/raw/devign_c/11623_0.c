static void test_io_channel_setup_async(SocketAddress *listen_addr,

                                        SocketAddress *connect_addr,

                                        QIOChannel **src,

                                        QIOChannel **dst)

{

    QIOChannelSocket *lioc;

    struct TestIOChannelData data;



    data.loop = g_main_loop_new(g_main_context_default(),

                                TRUE);



    lioc = qio_channel_socket_new();

    qio_channel_socket_listen_async(

        lioc, listen_addr,

        test_io_channel_complete, &data, NULL);



    g_main_loop_run(data.loop);

    g_main_context_iteration(g_main_context_default(), FALSE);



    g_assert(!data.err);



    if (listen_addr->type == SOCKET_ADDRESS_KIND_INET) {

        SocketAddress *laddr = qio_channel_socket_get_local_address(

            lioc, &error_abort);



        g_free(connect_addr->u.inet.data->port);

        connect_addr->u.inet.data->port = g_strdup(laddr->u.inet.data->port);



        qapi_free_SocketAddress(laddr);

    }



    *src = QIO_CHANNEL(qio_channel_socket_new());



    qio_channel_socket_connect_async(

        QIO_CHANNEL_SOCKET(*src), connect_addr,

        test_io_channel_complete, &data, NULL);



    g_main_loop_run(data.loop);

    g_main_context_iteration(g_main_context_default(), FALSE);



    g_assert(!data.err);



    qio_channel_wait(QIO_CHANNEL(lioc), G_IO_IN);

    *dst = QIO_CHANNEL(qio_channel_socket_accept(lioc, &error_abort));

    g_assert(*dst);



    qio_channel_set_delay(*src, false);

    test_io_channel_set_socket_bufs(*src, *dst);



    object_unref(OBJECT(lioc));



    g_main_loop_unref(data.loop);

}
