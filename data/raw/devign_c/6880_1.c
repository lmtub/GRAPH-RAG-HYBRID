static void tcp_chr_tls_init(CharDriverState *chr)

{

    TCPCharDriver *s = chr->opaque;

    QIOChannelTLS *tioc;

    Error *err = NULL;



    if (s->is_listen) {

        tioc = qio_channel_tls_new_server(

            s->ioc, s->tls_creds,

            NULL, /* XXX Use an ACL */

            &err);

    } else {

        tioc = qio_channel_tls_new_client(

            s->ioc, s->tls_creds,

            s->addr->u.inet.data->host,

            &err);

    }

    if (tioc == NULL) {

        error_free(err);

        tcp_chr_disconnect(chr);


    }

    object_unref(OBJECT(s->ioc));

    s->ioc = QIO_CHANNEL(tioc);



    qio_channel_tls_handshake(tioc,

                              tcp_chr_tls_handshake,

                              chr,

                              NULL);

}