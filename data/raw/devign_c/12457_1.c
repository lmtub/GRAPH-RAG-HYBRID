static void qio_channel_websock_handshake_send_res_ok(QIOChannelWebsock *ioc,

                                                      const char *key,

                                                      Error **errp)

{

    char combined_key[QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN +

                      QIO_CHANNEL_WEBSOCK_GUID_LEN + 1];

    char *accept = NULL;

    char *date = qio_channel_websock_date_str();



    g_strlcpy(combined_key, key, QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN + 1);

    g_strlcat(combined_key, QIO_CHANNEL_WEBSOCK_GUID,

              QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN +

              QIO_CHANNEL_WEBSOCK_GUID_LEN + 1);



    /* hash and encode it */

    if (qcrypto_hash_base64(QCRYPTO_HASH_ALG_SHA1,

                            combined_key,

                            QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN +

                            QIO_CHANNEL_WEBSOCK_GUID_LEN,

                            &accept,

                            errp) < 0) {

        qio_channel_websock_handshake_send_res_err(

            ioc, QIO_CHANNEL_WEBSOCK_HANDSHAKE_RES_SERVER_ERR);

        return;

    }



    qio_channel_websock_handshake_send_res(

        ioc, QIO_CHANNEL_WEBSOCK_HANDSHAKE_RES_OK, date, accept);



    g_free(date);

    g_free(accept);

}
