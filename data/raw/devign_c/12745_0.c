static int vnc_display_listen_addr(VncDisplay *vd,

                                   SocketAddressLegacy *addr,

                                   const char *name,

                                   QIOChannelSocket ***lsock,

                                   guint **lsock_tag,

                                   size_t *nlsock,

                                   Error **errp)

{

    QIODNSResolver *resolver = qio_dns_resolver_get_instance();

    SocketAddressLegacy **rawaddrs = NULL;

    size_t nrawaddrs = 0;

    Error *listenerr = NULL;

    bool listening = false;

    size_t i;



    if (qio_dns_resolver_lookup_sync(resolver, addr, &nrawaddrs,

                                     &rawaddrs, errp) < 0) {

        return -1;

    }



    for (i = 0; i < nrawaddrs; i++) {

        QIOChannelSocket *sioc = qio_channel_socket_new();



        qio_channel_set_name(QIO_CHANNEL(sioc), name);

        if (qio_channel_socket_listen_sync(

                sioc, rawaddrs[i], listenerr == NULL ? &listenerr : NULL) < 0) {

            object_unref(OBJECT(sioc));

            continue;

        }

        listening = true;

        (*nlsock)++;

        *lsock = g_renew(QIOChannelSocket *, *lsock, *nlsock);

        *lsock_tag = g_renew(guint, *lsock_tag, *nlsock);



        (*lsock)[*nlsock - 1] = sioc;

        (*lsock_tag)[*nlsock - 1] = 0;

    }



    for (i = 0; i < nrawaddrs; i++) {

        qapi_free_SocketAddressLegacy(rawaddrs[i]);

    }

    g_free(rawaddrs);



    if (listenerr) {

        if (!listening) {

            error_propagate(errp, listenerr);

            return -1;

        } else {

            error_free(listenerr);

        }

    }



    for (i = 0; i < *nlsock; i++) {

        (*lsock_tag)[i] = qio_channel_add_watch(

            QIO_CHANNEL((*lsock)[i]),

            G_IO_IN, vnc_listen_io, vd, NULL);

    }



    return 0;

}
