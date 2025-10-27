static int parse_uri(const char *filename, QDict *options, Error **errp)

{

    URI *uri = NULL;

    QueryParams *qp = NULL;

    int i;



    uri = uri_parse(filename);

    if (!uri) {

        return -EINVAL;

    }



    if (strcmp(uri->scheme, "ssh") != 0) {

        error_setg(errp, "URI scheme must be 'ssh'");

        goto err;

    }



    if (!uri->server || strcmp(uri->server, "") == 0) {

        error_setg(errp, "missing hostname in URI");

        goto err;

    }



    if (!uri->path || strcmp(uri->path, "") == 0) {

        error_setg(errp, "missing remote path in URI");

        goto err;

    }



    qp = query_params_parse(uri->query);

    if (!qp) {

        error_setg(errp, "could not parse query parameters");

        goto err;

    }



    if(uri->user && strcmp(uri->user, "") != 0) {

        qdict_put(options, "user", qstring_from_str(uri->user));

    }



    qdict_put(options, "host", qstring_from_str(uri->server));



    if (uri->port) {

        qdict_put(options, "port", qint_from_int(uri->port));

    }



    qdict_put(options, "path", qstring_from_str(uri->path));



    /* Pick out any query parameters that we understand, and ignore

     * the rest.

     */

    for (i = 0; i < qp->n; ++i) {

        if (strcmp(qp->p[i].name, "host_key_check") == 0) {

            qdict_put(options, "host_key_check",

                      qstring_from_str(qp->p[i].value));

        }

    }



    query_params_free(qp);

    uri_free(uri);

    return 0;



 err:

    if (qp) {

      query_params_free(qp);

    }

    if (uri) {

      uri_free(uri);

    }

    return -EINVAL;

}
