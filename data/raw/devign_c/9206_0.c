static void qio_channel_websock_handshake_process(QIOChannelWebsock *ioc,

                                                  char *buffer,

                                                  Error **errp)

{

    QIOChannelWebsockHTTPHeader hdrs[32];

    size_t nhdrs = G_N_ELEMENTS(hdrs);

    const char *protocols = NULL, *version = NULL, *key = NULL,

        *host = NULL, *connection = NULL, *upgrade = NULL;



    nhdrs = qio_channel_websock_extract_headers(ioc, buffer, hdrs, nhdrs, errp);

    if (!nhdrs) {

        return;

    }



    protocols = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_PROTOCOL);

    if (!protocols) {

        error_setg(errp, "Missing websocket protocol header data");

        goto bad_request;

    }



    version = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_VERSION);

    if (!version) {

        error_setg(errp, "Missing websocket version header data");

        goto bad_request;

    }



    key = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_KEY);

    if (!key) {

        error_setg(errp, "Missing websocket key header data");

        goto bad_request;

    }



    host = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_HOST);

    if (!host) {

        error_setg(errp, "Missing websocket host header data");

        goto bad_request;

    }



    connection = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_CONNECTION);

    if (!connection) {

        error_setg(errp, "Missing websocket connection header data");

        goto bad_request;

    }



    upgrade = qio_channel_websock_find_header(

        hdrs, nhdrs, QIO_CHANNEL_WEBSOCK_HEADER_UPGRADE);

    if (!upgrade) {

        error_setg(errp, "Missing websocket upgrade header data");

        goto bad_request;

    }



    if (!g_strrstr(protocols, QIO_CHANNEL_WEBSOCK_PROTOCOL_BINARY)) {

        error_setg(errp, "No '%s' protocol is supported by client '%s'",

                   QIO_CHANNEL_WEBSOCK_PROTOCOL_BINARY, protocols);

        goto bad_request;

    }



    if (!g_str_equal(version, QIO_CHANNEL_WEBSOCK_SUPPORTED_VERSION)) {

        error_setg(errp, "Version '%s' is not supported by client '%s'",

                   QIO_CHANNEL_WEBSOCK_SUPPORTED_VERSION, version);

        goto bad_request;

    }



    if (strlen(key) != QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN) {

        error_setg(errp, "Key length '%zu' was not as expected '%d'",

                   strlen(key), QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN);

        goto bad_request;

    }



    if (!g_strrstr(connection, QIO_CHANNEL_WEBSOCK_CONNECTION_UPGRADE)) {

        error_setg(errp, "No connection upgrade requested '%s'", connection);

        goto bad_request;

    }



    if (!g_str_equal(upgrade, QIO_CHANNEL_WEBSOCK_UPGRADE_WEBSOCKET)) {

        error_setg(errp, "Incorrect upgrade method '%s'", upgrade);

        goto bad_request;

    }



    qio_channel_websock_handshake_send_res_ok(ioc, key, errp);

    return;



 bad_request:

    qio_channel_websock_handshake_send_res_err(

        ioc, QIO_CHANNEL_WEBSOCK_HANDSHAKE_RES_BAD_REQUEST);

}
