static void test_io_channel_ipv4(bool async)

{

    SocketAddress *listen_addr = g_new0(SocketAddress, 1);

    SocketAddress *connect_addr = g_new0(SocketAddress, 1);



    listen_addr->type = SOCKET_ADDRESS_KIND_INET;

    listen_addr->u.inet = g_new0(InetSocketAddress, 1);

    listen_addr->u.inet->host = g_strdup("0.0.0.0");

    listen_addr->u.inet->port = NULL; /* Auto-select */



    connect_addr->type = SOCKET_ADDRESS_KIND_INET;

    connect_addr->u.inet = g_new0(InetSocketAddress, 1);

    connect_addr->u.inet->host = g_strdup("127.0.0.1");

    connect_addr->u.inet->port = NULL; /* Filled in later */



    test_io_channel(async, listen_addr, connect_addr);



    qapi_free_SocketAddress(listen_addr);

    qapi_free_SocketAddress(connect_addr);

}
