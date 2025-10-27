static void nbd_refresh_filename(BlockDriverState *bs, QDict *options)

{

    BDRVNBDState *s = bs->opaque;

    QDict *opts = qdict_new();

    QObject *saddr_qdict;

    Visitor *ov;

    const char *host = NULL, *port = NULL, *path = NULL;



    if (s->saddr->type == SOCKET_ADDRESS_KIND_INET) {

        const InetSocketAddress *inet = s->saddr->u.inet.data;

        if (!inet->has_ipv4 && !inet->has_ipv6 && !inet->has_to) {

            host = inet->host;

            port = inet->port;

        }

    } else if (s->saddr->type == SOCKET_ADDRESS_KIND_UNIX) {

        path = s->saddr->u.q_unix.data->path;

    }



    qdict_put(opts, "driver", qstring_from_str("nbd"));



    if (path && s->export) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd+unix:///%s?socket=%s", s->export, path);

    } else if (path && !s->export) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd+unix://?socket=%s", path);

    } else if (host && s->export) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd://%s:%s/%s", host, port, s->export);

    } else if (host && !s->export) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd://%s:%s", host, port);

    }



    ov = qobject_output_visitor_new(&saddr_qdict);

    visit_type_SocketAddress(ov, NULL, &s->saddr, &error_abort);

    visit_complete(ov, &saddr_qdict);


    assert(qobject_type(saddr_qdict) == QTYPE_QDICT);



    qdict_put_obj(opts, "server", saddr_qdict);



    if (s->export) {

        qdict_put(opts, "export", qstring_from_str(s->export));

    }

    if (s->tlscredsid) {

        qdict_put(opts, "tls-creds", qstring_from_str(s->tlscredsid));

    }



    qdict_flatten(opts);

    bs->full_open_options = opts;

}