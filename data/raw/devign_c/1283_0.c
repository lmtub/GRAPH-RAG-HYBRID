static CharDriverState *qmp_chardev_open_socket(const char *id,

                                                ChardevBackend *backend,

                                                ChardevReturn *ret,

                                                Error **errp)

{

    CharDriverState *chr;

    TCPCharDriver *s;

    ChardevSocket *sock = backend->u.socket;

    SocketAddress *addr = sock->addr;

    bool do_nodelay     = sock->has_nodelay ? sock->nodelay : false;

    bool is_listen      = sock->has_server  ? sock->server  : true;

    bool is_telnet      = sock->has_telnet  ? sock->telnet  : false;

    bool is_waitconnect = sock->has_wait    ? sock->wait    : false;

    int64_t reconnect   = sock->has_reconnect ? sock->reconnect : 0;

    ChardevCommon *common = qapi_ChardevSocket_base(sock);

    QIOChannelSocket *sioc = NULL;



    chr = qemu_chr_alloc(common, errp);

    if (!chr) {

        return NULL;

    }

    s = g_new0(TCPCharDriver, 1);



    s->is_unix = addr->type == SOCKET_ADDRESS_KIND_UNIX;

    s->is_listen = is_listen;

    s->is_telnet = is_telnet;

    s->do_nodelay = do_nodelay;

    if (sock->tls_creds) {

        Object *creds;

        creds = object_resolve_path_component(

            object_get_objects_root(), sock->tls_creds);

        if (!creds) {

            error_setg(errp, "No TLS credentials with id '%s'",

                       sock->tls_creds);

            goto error;

        }

        s->tls_creds = (QCryptoTLSCreds *)

            object_dynamic_cast(creds,

                                TYPE_QCRYPTO_TLS_CREDS);

        if (!s->tls_creds) {

            error_setg(errp, "Object with id '%s' is not TLS credentials",

                       sock->tls_creds);

            goto error;

        }

        object_ref(OBJECT(s->tls_creds));

        if (is_listen) {

            if (s->tls_creds->endpoint != QCRYPTO_TLS_CREDS_ENDPOINT_SERVER) {

                error_setg(errp, "%s",

                           "Expected TLS credentials for server endpoint");

                goto error;

            }

        } else {

            if (s->tls_creds->endpoint != QCRYPTO_TLS_CREDS_ENDPOINT_CLIENT) {

                error_setg(errp, "%s",

                           "Expected TLS credentials for client endpoint");

                goto error;

            }

        }

    }



    qapi_copy_SocketAddress(&s->addr, sock->addr);



    chr->opaque = s;

    chr->chr_write = tcp_chr_write;

    chr->chr_sync_read = tcp_chr_sync_read;

    chr->chr_close = tcp_chr_close;

    chr->get_msgfds = tcp_get_msgfds;

    chr->set_msgfds = tcp_set_msgfds;

    chr->chr_add_client = tcp_chr_add_client;

    chr->chr_add_watch = tcp_chr_add_watch;

    chr->chr_update_read_handler = tcp_chr_update_read_handler;

    /* be isn't opened until we get a connection */

    chr->explicit_be_open = true;



    chr->filename = SocketAddress_to_str("disconnected:",

                                         addr, is_listen, is_telnet);



    if (is_listen) {

        if (is_telnet) {

            s->do_telnetopt = 1;

        }

    } else if (reconnect > 0) {

        s->reconnect_time = reconnect;

    }



    sioc = qio_channel_socket_new();

    if (s->reconnect_time) {

        qio_channel_socket_connect_async(sioc, s->addr,

                                         qemu_chr_socket_connected,

                                         chr, NULL);

    } else if (s->is_listen) {

        if (qio_channel_socket_listen_sync(sioc, s->addr, errp) < 0) {

            goto error;

        }

        s->listen_ioc = sioc;

        if (is_waitconnect) {

            fprintf(stderr, "QEMU waiting for connection on: %s\n",

                    chr->filename);

            tcp_chr_accept(QIO_CHANNEL(s->listen_ioc), G_IO_IN, chr);

        }

        qio_channel_set_blocking(QIO_CHANNEL(s->listen_ioc), false, NULL);

        if (!s->ioc) {

            s->listen_tag = qio_channel_add_watch(

                QIO_CHANNEL(s->listen_ioc), G_IO_IN, tcp_chr_accept, chr, NULL);

        }

    } else {

        if (qio_channel_socket_connect_sync(sioc, s->addr, errp) < 0) {

            goto error;

        }

        tcp_chr_new_client(chr, sioc);

        object_unref(OBJECT(sioc));

    }



    return chr;



 error:

    if (sioc) {

        object_unref(OBJECT(sioc));

    }

    if (s->tls_creds) {

        object_unref(OBJECT(s->tls_creds));

    }

    g_free(s);

    qemu_chr_free_common(chr);

    return NULL;

}
