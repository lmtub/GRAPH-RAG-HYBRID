static int curl_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    BDRVCURLState *s = bs->opaque;

    CURLState *state = NULL;

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *file;

    double d;



    static int inited = 0;



    if (flags & BDRV_O_RDWR) {

        error_setg(errp, "curl block device does not support writes");

        return -EROFS;

    }



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out_noclean;

    }



    s->readahead_size = qemu_opt_get_size(opts, "readahead", READ_AHEAD_SIZE);

    if ((s->readahead_size & 0x1ff) != 0) {

        error_setg(errp, "HTTP_READAHEAD_SIZE %zd is not a multiple of 512",

                   s->readahead_size);

        goto out_noclean;

    }



    file = qemu_opt_get(opts, "url");

    if (file == NULL) {

        error_setg(errp, "curl block driver requires an 'url' option");

        goto out_noclean;

    }



    if (!inited) {

        curl_global_init(CURL_GLOBAL_ALL);

        inited = 1;

    }



    DPRINTF("CURL: Opening %s\n", file);

    s->url = g_strdup(file);

    state = curl_init_state(s);

    if (!state)

        goto out_noclean;



    // Get file size



    s->accept_range = false;

    curl_easy_setopt(state->curl, CURLOPT_NOBODY, 1);

    curl_easy_setopt(state->curl, CURLOPT_HEADERFUNCTION,

                     curl_header_cb);

    curl_easy_setopt(state->curl, CURLOPT_HEADERDATA, s);

    if (curl_easy_perform(state->curl))

        goto out;

    curl_easy_getinfo(state->curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &d);

    if (d)

        s->len = (size_t)d;

    else if(!s->len)

        goto out;

    if ((!strncasecmp(s->url, "http://", strlen("http://"))

        || !strncasecmp(s->url, "https://", strlen("https://")))

        && !s->accept_range) {

        pstrcpy(state->errmsg, CURL_ERROR_SIZE,

                "Server does not support 'range' (byte ranges).");

        goto out;

    }

    DPRINTF("CURL: Size = %zd\n", s->len);



    curl_clean_state(state);

    curl_easy_cleanup(state->curl);

    state->curl = NULL;



    aio_timer_init(bdrv_get_aio_context(bs), &s->timer,

                   QEMU_CLOCK_REALTIME, SCALE_NS,

                   curl_multi_timeout_do, s);



    // Now we know the file exists and its size, so let's

    // initialize the multi interface!



    s->multi = curl_multi_init();

    curl_multi_setopt(s->multi, CURLMOPT_SOCKETDATA, s);

    curl_multi_setopt(s->multi, CURLMOPT_SOCKETFUNCTION, curl_sock_cb);

#ifdef NEED_CURL_TIMER_CALLBACK

    curl_multi_setopt(s->multi, CURLMOPT_TIMERDATA, s);

    curl_multi_setopt(s->multi, CURLMOPT_TIMERFUNCTION, curl_timer_cb);

#endif



    qemu_opts_del(opts);

    return 0;



out:

    error_setg(errp, "CURL: Error opening file: %s", state->errmsg);

    curl_easy_cleanup(state->curl);

    state->curl = NULL;

out_noclean:

    g_free(s->url);

    qemu_opts_del(opts);

    return -EINVAL;

}
