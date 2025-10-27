static void test_io_channel_tls(const void *opaque)

{

    struct QIOChannelTLSTestData *data =

        (struct QIOChannelTLSTestData *)opaque;

    QCryptoTLSCreds *clientCreds;

    QCryptoTLSCreds *serverCreds;

    QIOChannelTLS *clientChanTLS;

    QIOChannelTLS *serverChanTLS;

    QIOChannelSocket *clientChanSock;

    QIOChannelSocket *serverChanSock;

    qemu_acl *acl;

    const char * const *wildcards;

    int channel[2];

    struct QIOChannelTLSHandshakeData clientHandshake = { false, false };

    struct QIOChannelTLSHandshakeData serverHandshake = { false, false };

    Error *err = NULL;

    QIOChannelTest *test;

    GMainContext *mainloop;



    /* We'll use this for our fake client-server connection */

    g_assert(socketpair(AF_UNIX, SOCK_STREAM, 0, channel) == 0);



#define CLIENT_CERT_DIR "tests/test-io-channel-tls-client/"

#define SERVER_CERT_DIR "tests/test-io-channel-tls-server/"

    mkdir(CLIENT_CERT_DIR, 0700);

    mkdir(SERVER_CERT_DIR, 0700);



    unlink(SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_CA_CERT);

    unlink(SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_SERVER_CERT);

    unlink(SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_SERVER_KEY);



    unlink(CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CA_CERT);

    unlink(CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CLIENT_CERT);

    unlink(CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CLIENT_KEY);



    g_assert(link(data->servercacrt,

                  SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_CA_CERT) == 0);

    g_assert(link(data->servercrt,

                  SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_SERVER_CERT) == 0);

    g_assert(link(KEYFILE,

                  SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_SERVER_KEY) == 0);



    g_assert(link(data->clientcacrt,

                  CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CA_CERT) == 0);

    g_assert(link(data->clientcrt,

                  CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CLIENT_CERT) == 0);

    g_assert(link(KEYFILE,

                  CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CLIENT_KEY) == 0);



    clientCreds = test_tls_creds_create(

        QCRYPTO_TLS_CREDS_ENDPOINT_CLIENT,

        CLIENT_CERT_DIR,

        &err);

    g_assert(clientCreds != NULL);



    serverCreds = test_tls_creds_create(

        QCRYPTO_TLS_CREDS_ENDPOINT_SERVER,

        SERVER_CERT_DIR,

        &err);

    g_assert(serverCreds != NULL);



    acl = qemu_acl_init("channeltlsacl");

    qemu_acl_reset(acl);

    wildcards = data->wildcards;

    while (wildcards && *wildcards) {

        qemu_acl_append(acl, 0, *wildcards);

        wildcards++;

    }



    clientChanSock = qio_channel_socket_new_fd(

        channel[0], &err);

    g_assert(clientChanSock != NULL);

    serverChanSock = qio_channel_socket_new_fd(

        channel[1], &err);

    g_assert(serverChanSock != NULL);



    /*

     * We have an evil loop to do the handshake in a single

     * thread, so we need these non-blocking to avoid deadlock

     * of ourselves

     */

    qio_channel_set_blocking(QIO_CHANNEL(clientChanSock), false, NULL);

    qio_channel_set_blocking(QIO_CHANNEL(serverChanSock), false, NULL);



    /* Now the real part of the test, setup the sessions */

    clientChanTLS = qio_channel_tls_new_client(

        QIO_CHANNEL(clientChanSock), clientCreds,

        data->hostname, &err);

    g_assert(clientChanTLS != NULL);



    serverChanTLS = qio_channel_tls_new_server(

        QIO_CHANNEL(serverChanSock), serverCreds,

        "channeltlsacl", &err);

    g_assert(serverChanTLS != NULL);



    qio_channel_tls_handshake(clientChanTLS,

                              test_tls_handshake_done,

                              &clientHandshake,

                              NULL);

    qio_channel_tls_handshake(serverChanTLS,

                              test_tls_handshake_done,

                              &serverHandshake,

                              NULL);



    /*

     * Finally we loop around & around doing handshake on each

     * session until we get an error, or the handshake completes.

     * This relies on the socketpair being nonblocking to avoid

     * deadlocking ourselves upon handshake

     */

    mainloop = g_main_context_default();

    do {

        g_main_context_iteration(mainloop, TRUE);

    } while (!clientHandshake.finished &&

             !serverHandshake.finished);



    g_assert(clientHandshake.failed == data->expectClientFail);

    g_assert(serverHandshake.failed == data->expectServerFail);



    test = qio_channel_test_new();

    qio_channel_test_run_threads(test, false,

                                 QIO_CHANNEL(clientChanTLS),

                                 QIO_CHANNEL(serverChanTLS));

    qio_channel_test_validate(test);



    test = qio_channel_test_new();

    qio_channel_test_run_threads(test, true,

                                 QIO_CHANNEL(clientChanTLS),

                                 QIO_CHANNEL(serverChanTLS));

    qio_channel_test_validate(test);



    unlink(SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_CA_CERT);

    unlink(SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_SERVER_CERT);

    unlink(SERVER_CERT_DIR QCRYPTO_TLS_CREDS_X509_SERVER_KEY);



    unlink(CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CA_CERT);

    unlink(CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CLIENT_CERT);

    unlink(CLIENT_CERT_DIR QCRYPTO_TLS_CREDS_X509_CLIENT_KEY);



    rmdir(CLIENT_CERT_DIR);

    rmdir(SERVER_CERT_DIR);



    object_unparent(OBJECT(serverCreds));

    object_unparent(OBJECT(clientCreds));



    object_unref(OBJECT(serverChanTLS));

    object_unref(OBJECT(clientChanTLS));



    object_unref(OBJECT(serverChanSock));

    object_unref(OBJECT(clientChanSock));



    close(channel[0]);

    close(channel[1]);

}
