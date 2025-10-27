static int sd_parse_uri(BDRVSheepdogState *s, const char *filename,

                        char *vdi, uint32_t *snapid, char *tag)

{

    URI *uri;

    QueryParams *qp = NULL;

    int ret = 0;



    uri = uri_parse(filename);

    if (!uri) {

        return -EINVAL;

    }



    /* transport */

    if (!strcmp(uri->scheme, "sheepdog")) {

        s->is_unix = false;

    } else if (!strcmp(uri->scheme, "sheepdog+tcp")) {

        s->is_unix = false;

    } else if (!strcmp(uri->scheme, "sheepdog+unix")) {

        s->is_unix = true;

    } else {

        ret = -EINVAL;

        goto out;

    }



    if (uri->path == NULL || !strcmp(uri->path, "/")) {

        ret = -EINVAL;

        goto out;

    }

    pstrcpy(vdi, SD_MAX_VDI_LEN, uri->path + 1);



    qp = query_params_parse(uri->query);

    if (qp->n > 1 || (s->is_unix && !qp->n) || (!s->is_unix && qp->n)) {

        ret = -EINVAL;

        goto out;

    }



    if (s->is_unix) {

        /* sheepdog+unix:///vdiname?socket=path */

        if (uri->server || uri->port || strcmp(qp->p[0].name, "socket")) {

            ret = -EINVAL;

            goto out;

        }

        s->host_spec = g_strdup(qp->p[0].value);

    } else {

        /* sheepdog[+tcp]://[host:port]/vdiname */

        s->host_spec = g_strdup_printf("%s:%d", uri->server ?: SD_DEFAULT_ADDR,

                                       uri->port ?: SD_DEFAULT_PORT);

    }



    /* snapshot tag */

    if (uri->fragment) {

        *snapid = strtoul(uri->fragment, NULL, 10);

        if (*snapid == 0) {

            pstrcpy(tag, SD_MAX_VDI_TAG_LEN, uri->fragment);

        }

    } else {

        *snapid = CURRENT_VDI_ID; /* search current vdi */

    }



out:

    if (qp) {

        query_params_free(qp);

    }

    uri_free(uri);

    return ret;

}
