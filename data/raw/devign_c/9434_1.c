static int64_t nfs_client_open(NFSClient *client, const char *filename,

                               int flags, Error **errp)

{

    int ret = -EINVAL, i;

    struct stat st;

    URI *uri;

    QueryParams *qp = NULL;

    char *file = NULL, *strp = NULL;



    uri = uri_parse(filename);

    if (!uri) {

        error_setg(errp, "Invalid URL specified");

        goto fail;

    }

    if (!uri->server) {

        error_setg(errp, "Invalid URL specified");

        goto fail;

    }

    strp = strrchr(uri->path, '/');

    if (strp == NULL) {

        error_setg(errp, "Invalid URL specified");

        goto fail;

    }

    file = g_strdup(strp);

    *strp = 0;



    client->context = nfs_init_context();

    if (client->context == NULL) {

        error_setg(errp, "Failed to init NFS context");

        goto fail;

    }



    qp = query_params_parse(uri->query);

    for (i = 0; i < qp->n; i++) {

        if (!qp->p[i].value) {

            error_setg(errp, "Value for NFS parameter expected: %s",

                       qp->p[i].name);

            goto fail;

        }

        if (!strncmp(qp->p[i].name, "uid", 3)) {

            nfs_set_uid(client->context, atoi(qp->p[i].value));

        } else if (!strncmp(qp->p[i].name, "gid", 3)) {

            nfs_set_gid(client->context, atoi(qp->p[i].value));

        } else if (!strncmp(qp->p[i].name, "tcp-syncnt", 10)) {

            nfs_set_tcp_syncnt(client->context, atoi(qp->p[i].value));

        } else {

            error_setg(errp, "Unknown NFS parameter name: %s",

                       qp->p[i].name);

            goto fail;

        }

    }



    ret = nfs_mount(client->context, uri->server, uri->path);

    if (ret < 0) {

        error_setg(errp, "Failed to mount nfs share: %s",

                   nfs_get_error(client->context));

        goto fail;

    }



    if (flags & O_CREAT) {

        ret = nfs_creat(client->context, file, 0600, &client->fh);

        if (ret < 0) {

            error_setg(errp, "Failed to create file: %s",

                       nfs_get_error(client->context));

            goto fail;

        }

    } else {

        ret = nfs_open(client->context, file, flags, &client->fh);

        if (ret < 0) {

            error_setg(errp, "Failed to open file : %s",

                       nfs_get_error(client->context));

            goto fail;

        }

    }



    ret = nfs_fstat(client->context, client->fh, &st);

    if (ret < 0) {

        error_setg(errp, "Failed to fstat file: %s",

                   nfs_get_error(client->context));

        goto fail;

    }



    ret = DIV_ROUND_UP(st.st_size, BDRV_SECTOR_SIZE);

    client->has_zero_init = S_ISREG(st.st_mode);

    goto out;

fail:

    nfs_client_close(client);

out:

    if (qp) {

        query_params_free(qp);

    }

    uri_free(uri);

    g_free(file);

    return ret;

}
