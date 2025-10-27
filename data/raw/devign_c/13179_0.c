static void test_io_channel_ipv6(bool async)

{

    SocketAddress *listen_addr = g_new0(SocketAddress, 1);

    SocketAddress *connect_addr = g_new0(SocketAddress, 1);



    listen_addr->type = SOCKET_ADDRESS_KIND_INET;

    listen_addr->u.inet = g_new(InetSocketAddress, 1);

    *listen_addr->u.inet = (InetSocketAddress) {

        .host = g_strdup("::1"),

        .port = NULL, /* Auto-select */

    };



    connect_addr->type = SOCKET_ADDRESS_KIND_INET;

    connect_addr->u.inet = g_new(InetSocketAddress, 1);

    *connect_addr->u.inet = (InetSocketAddress) {

        .host = g_strdup("::1"),

        .port = NULL, /* Filled in later */

    };



    test_io_channel(async, listen_addr, connect_addr, false);



    qapi_free_SocketAddress(listen_addr);

    qapi_free_SocketAddress(connect_addr);

}
