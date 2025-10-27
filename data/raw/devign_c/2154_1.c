static int curl_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    BDRVCURLState *s = bs->opaque;

    CURLState *state = NULL;

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *file;

    const char *cookie;

    const char *cookie_secret;

    double d;

    const char *secretid;

    const char *protocol_delimiter;



    static int inited = 0;



    if (flags & BDRV_O_RDWR) {

        error_setg(errp, "curl block device does not support writes");

        return -EROFS;

    }



    qemu_mutex_init(&s->mutex);

    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out_noclean;

    }



    s->readahead_size = qemu_opt_get_size(opts, CURL_BLOCK_OPT_READAHEAD,

                                          READ_AHEAD_DEFAULT);

    if ((s->readahead_size & 0x1ff) != 0) {

        error_setg(errp, "HTTP_READAHEAD_SIZE %zd is not a multiple of 512",

                   s->readahead_size);

        goto out_noclean;

    }



    s->timeout = qemu_opt_get_number(opts, CURL_BLOCK_OPT_TIMEOUT,

                                     CURL_TIMEOUT_DEFAULT);

    if (s->timeout > CURL_TIMEOUT_MAX) {

        error_setg(errp, "timeout parameter is too large or negative");

        goto out_noclean;

    }



    s->sslverify = qemu_opt_get_bool(opts, CURL_BLOCK_OPT_SSLVERIFY, true);



    cookie = qemu_opt_get(opts, CURL_BLOCK_OPT_COOKIE);

    cookie_secret = qemu_opt_get(opts, CURL_BLOCK_OPT_COOKIE_SECRET);



    if (cookie && cookie_secret) {

        error_setg(errp,

                   "curl driver cannot handle both cookie and cookie secret");

        goto out_noclean;

    }



    if (cookie_secret) {

        s->cookie = qcrypto_secret_lookup_as_utf8(cookie_secret, errp);

        if (!s->cookie) {

            goto out_noclean;

        }

    } else {

        s->cookie = g_strdup(cookie);

    }



    file = qemu_opt_get(opts, CURL_BLOCK_OPT_URL);

    if (file == NULL) {

        error_setg(errp, "curl block driver requires an 'url' option");

        goto out_noclean;

    }



    if (!strstart(file, bs->drv->protocol_name, &protocol_delimiter) ||

        !strstart(protocol_delimiter, "://", NULL))

    {

        error_setg(errp, "%s curl driver cannot handle the URL '%s' (does not "

                   "start with '%s://')", bs->drv->protocol_name, file,

                   bs->drv->protocol_name);

        goto out_noclean;

    }



    s->username = g_strdup(qemu_opt_get(opts, CURL_BLOCK_OPT_USERNAME));

    secretid = qemu_opt_get(opts, CURL_BLOCK_OPT_PASSWORD_SECRET);



    if (secretid) {

        s->password = qcrypto_secret_lookup_as_utf8(secretid, errp);

        if (!s->password) {

            goto out_noclean;

        }

    }



    s->proxyusername = g_strdup(

        qemu_opt_get(opts, CURL_BLOCK_OPT_PROXY_USERNAME));

    secretid = qemu_opt_get(opts, CURL_BLOCK_OPT_PROXY_PASSWORD_SECRET);

    if (secretid) {

        s->proxypassword = qcrypto_secret_lookup_as_utf8(secretid, errp);

        if (!s->proxypassword) {

            goto out_noclean;

        }

    }



    if (!inited) {

        curl_global_init(CURL_GLOBAL_ALL);

        inited = 1;

    }



    DPRINTF("CURL: Opening %s\n", file);

    QSIMPLEQ_INIT(&s->free_state_waitq);

    s->aio_context = bdrv_get_aio_context(bs);

    s->url = g_strdup(file);

    qemu_mutex_lock(&s->mutex);

    state = curl_find_state(s);

    qemu_mutex_unlock(&s->mutex);

    if (!state) {

        goto out_noclean;

    }



    // Get file size



    if (curl_init_state(s, state) < 0) {

        goto out;

    }



    s->accept_range = false;

    curl_easy_setopt(state->curl, CURLOPT_NOBODY, 1);

    curl_easy_setopt(state->curl, CURLOPT_HEADERFUNCTION,

                     curl_header_cb);

    curl_easy_setopt(state->curl, CURLOPT_HEADERDATA, s);

    if (curl_easy_perform(state->curl))

        goto out;

    if (curl_easy_getinfo(state->curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &d)) {

        goto out;

    }

    /* Prior CURL 7.19.4 return value of 0 could mean that the file size is not

     * know or the size is zero. From 7.19.4 CURL returns -1 if size is not

     * known and zero if it is realy zero-length file. */

#if LIBCURL_VERSION_NUM >= 0x071304

    if (d < 0) {

        pstrcpy(state->errmsg, CURL_ERROR_SIZE,

                "Server didn't report file size.");

        goto out;

    }

#else

    if (d <= 0) {

        pstrcpy(state->errmsg, CURL_ERROR_SIZE,

                "Unknown file size or zero-length file.");

        goto out;

    }

#endif



    s->len = d;



    if ((!strncasecmp(s->url, "http://", strlen("http://"))

        || !strncasecmp(s->url, "https://", strlen("https://")))

        && !s->accept_range) {

        pstrcpy(state->errmsg, CURL_ERROR_SIZE,

                "Server does not support 'range' (byte ranges).");

        goto out;

    }

    DPRINTF("CURL: Size = %" PRIu64 "\n", s->len);



    qemu_mutex_lock(&s->mutex);

    curl_clean_state(state);

    qemu_mutex_unlock(&s->mutex);

    curl_easy_cleanup(state->curl);

    state->curl = NULL;



    curl_attach_aio_context(bs, bdrv_get_aio_context(bs));



    qemu_opts_del(opts);

    return 0;



out:

    error_setg(errp, "CURL: Error opening file: %s", state->errmsg);

    curl_easy_cleanup(state->curl);

    state->curl = NULL;

out_noclean:

    qemu_mutex_destroy(&s->mutex);

    g_free(s->cookie);

    g_free(s->url);

    qemu_opts_del(opts);

    return -EINVAL;

}
