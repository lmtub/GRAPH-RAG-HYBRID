static int qio_channel_websock_handshake_process(QIOChannelWebsock *ioc,

                                                 const char *line,

                                                 size_t size,

                                                 Error **errp)

{

    int ret = -1;

    char *protocols = qio_channel_websock_handshake_entry(

        line, size, QIO_CHANNEL_WEBSOCK_HEADER_PROTOCOL);

    char *version = qio_channel_websock_handshake_entry(

        line, size, QIO_CHANNEL_WEBSOCK_HEADER_VERSION);

    char *key = qio_channel_websock_handshake_entry(

        line, size, QIO_CHANNEL_WEBSOCK_HEADER_KEY);



    if (!protocols) {

        error_setg(errp, "Missing websocket protocol header data");

        goto cleanup;

    }



    if (!version) {

        error_setg(errp, "Missing websocket version header data");

        goto cleanup;

    }



    if (!key) {

        error_setg(errp, "Missing websocket key header data");

        goto cleanup;

    }



    if (!g_strrstr(protocols, QIO_CHANNEL_WEBSOCK_PROTOCOL_BINARY)) {

        error_setg(errp, "No '%s' protocol is supported by client '%s'",

                   QIO_CHANNEL_WEBSOCK_PROTOCOL_BINARY, protocols);

        goto cleanup;

    }



    if (!g_str_equal(version, QIO_CHANNEL_WEBSOCK_SUPPORTED_VERSION)) {

        error_setg(errp, "Version '%s' is not supported by client '%s'",

                   QIO_CHANNEL_WEBSOCK_SUPPORTED_VERSION, version);

        goto cleanup;

    }



    if (strlen(key) != QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN) {

        error_setg(errp, "Key length '%zu' was not as expected '%d'",

                   strlen(key), QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN);

        goto cleanup;

    }



    ret = qio_channel_websock_handshake_send_response(ioc, key, errp);



 cleanup:

    g_free(protocols);

    g_free(version);

    g_free(key);

    return ret;

}
