static void test_io_channel_unix(bool async)

{

    SocketAddress *listen_addr = g_new0(SocketAddress, 1);

    SocketAddress *connect_addr = g_new0(SocketAddress, 1);



#define TEST_SOCKET "test-io-channel-socket.sock"

    listen_addr->type = SOCKET_ADDRESS_KIND_UNIX;

    listen_addr->u.q_unix = g_new0(UnixSocketAddress, 1);

    listen_addr->u.q_unix->path = g_strdup(TEST_SOCKET);



    connect_addr->type = SOCKET_ADDRESS_KIND_UNIX;

    connect_addr->u.q_unix = g_new0(UnixSocketAddress, 1);

    connect_addr->u.q_unix->path = g_strdup(TEST_SOCKET);



    test_io_channel(async, listen_addr, connect_addr, true);



    qapi_free_SocketAddress(listen_addr);

    qapi_free_SocketAddress(connect_addr);

    unlink(TEST_SOCKET);

}
