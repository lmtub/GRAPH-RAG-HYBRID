static int qio_channel_websock_handshake_process(QIOChannelWebsock *ioc,

                                                 char *buffer,

                                                 Error **errp)

{

    QIOChannelWebsockHTTPHeader hdrs[32];

    size_t nhdrs = G_N_ELEMENTS(hdrs);

    const char *protocols = NULL, *version = NULL, *key = NULL,

        *host = NULL, *connection = NULL, *upgrade = NULL;



    nhdrs = qio_channel_websock_extract_headers(buffer, hdrs, nhdrs, errp);

    if (!nhdrs) {

        return -1;

    }



    protocols = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_PROTOCOL);

    if (!protocols) {

        error_setg(errp, "Missing websocket protocol header data");

        return -1;

    }



    version = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_VERSION);

    if (!version) {

        error_setg(errp, "Missing websocket version header data");

        return -1;

    }



    key = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_KEY);

    if (!key) {

        error_setg(errp, "Missing websocket key header data");

        return -1;

    }



    host = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_HOST);

    if (!host) {

        error_setg(errp, "Missing websocket host header data");

        return -1;

    }



    connection = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_CONNECTION);

    if (!connection) {

        error_setg(errp, "Missing websocket connection header data");

        return -1;

    }



    upgrade = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_UPGRADE);

    if (!upgrade) {

        error_setg(errp, "Missing websocket upgrade header data");

        return -1;

    }



    if (!g_strrstr(protocols, QIO_CHANNEL_WEBSOCK_PROTOCOL_BINARY)) {

        error_setg(errp, "No '%s' protocol is supported by client '%s'",

                   QIO_CHANNEL_WEBSOCK_PROTOCOL_BINARY, protocols);

        return -1;

    }



    if (!g_str_equal(version, QIO_CHANNEL_WEBSOCK_SUPPORTED_VERSION)) {

        error_setg(errp, "Version '%s' is not supported by client '%s'",

                   QIO_CHANNEL_WEBSOCK_SUPPORTED_VERSION, version);

        return -1;

    }



    if (strlen(key) != QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN) {

        error_setg(errp, "Key length '%zu' was not as expected '%d'",

                   strlen(key), QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN);

        return -1;

    }



    if (!g_strrstr(connection, QIO_CHANNEL_WEBSOCK_CONNECTION_UPGRADE)) {

        error_setg(errp, "No connection upgrade requested '%s'", connection);

        return -1;

    }



    if (!g_str_equal(upgrade, QIO_CHANNEL_WEBSOCK_UPGRADE_WEBSOCKET)) {

        error_setg(errp, "Incorrect upgrade method '%s'", upgrade);

        return -1;

    }



    return qio_channel_websock_handshake_send_response(ioc, key, errp);

}
