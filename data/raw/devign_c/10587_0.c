static void tcp_chr_tls_init(Chardev *chr)

{

    SocketChardev *s = SOCKET_CHARDEV(chr);

    QIOChannelTLS *tioc;

    Error *err = NULL;

    gchar *name;



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

        return;

    }

    name = g_strdup_printf("chardev-tls-%s-%s",

                           s->is_listen ? "server" : "client",

                           chr->label);

    qio_channel_set_name(QIO_CHANNEL(tioc), name);

    g_free(name);

    object_unref(OBJECT(s->ioc));

    s->ioc = QIO_CHANNEL(tioc);



    qio_channel_tls_handshake(tioc,

                              tcp_chr_tls_handshake,

                              chr,

                              NULL);

}
