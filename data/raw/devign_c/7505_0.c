static int nbd_parse_uri(const char *filename, QDict *options)

{

    URI *uri;

    const char *p;

    QueryParams *qp = NULL;

    int ret = 0;

    bool is_unix;



    uri = uri_parse(filename);

    if (!uri) {

        return -EINVAL;

    }



    /* transport */

    if (!strcmp(uri->scheme, "nbd")) {

        is_unix = false;

    } else if (!strcmp(uri->scheme, "nbd+tcp")) {

        is_unix = false;

    } else if (!strcmp(uri->scheme, "nbd+unix")) {

        is_unix = true;

    } else {

        ret = -EINVAL;

        goto out;

    }



    p = uri->path ? uri->path : "/";

    p += strspn(p, "/");

    if (p[0]) {

        qdict_put(options, "export", qstring_from_str(p));

    }



    qp = query_params_parse(uri->query);

    if (qp->n > 1 || (is_unix && !qp->n) || (!is_unix && qp->n)) {

        ret = -EINVAL;

        goto out;

    }



    if (is_unix) {

        /* nbd+unix:///export?socket=path */

        if (uri->server || uri->port || strcmp(qp->p[0].name, "socket")) {

            ret = -EINVAL;

            goto out;

        }

        qdict_put(options, "server.type", qstring_from_str("unix"));

        qdict_put(options, "server.data.path",

                  qstring_from_str(qp->p[0].value));

    } else {

        QString *host;

        char *port_str;



        /* nbd[+tcp]://host[:port]/export */

        if (!uri->server) {

            ret = -EINVAL;

            goto out;

        }



        /* strip braces from literal IPv6 address */

        if (uri->server[0] == '[') {

            host = qstring_from_substr(uri->server, 1,

                                       strlen(uri->server) - 2);

        } else {

            host = qstring_from_str(uri->server);

        }



        qdict_put(options, "server.type", qstring_from_str("inet"));

        qdict_put(options, "server.data.host", host);



        port_str = g_strdup_printf("%d", uri->port ?: NBD_DEFAULT_PORT);

        qdict_put(options, "server.data.port", qstring_from_str(port_str));

        g_free(port_str);

    }



out:

    if (qp) {

        query_params_free(qp);

    }

    uri_free(uri);

    return ret;

}
