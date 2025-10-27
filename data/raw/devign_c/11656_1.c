static int qemu_rbd_open(BlockDriverState *bs, QDict *options, int flags)

{

    BDRVRBDState *s = bs->opaque;

    char pool[RBD_MAX_POOL_NAME_SIZE];

    char snap_buf[RBD_MAX_SNAP_NAME_SIZE];

    char conf[RBD_MAX_CONF_SIZE];

    char clientname_buf[RBD_MAX_CONF_SIZE];

    char *clientname;

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *filename;

    int r;



    opts = qemu_opts_create_nofail(&runtime_opts);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (error_is_set(&local_err)) {

        qerror_report_err(local_err);

        error_free(local_err);

        qemu_opts_del(opts);

        return -EINVAL;

    }



    filename = qemu_opt_get(opts, "filename");

    qemu_opts_del(opts);



    if (qemu_rbd_parsename(filename, pool, sizeof(pool),

                           snap_buf, sizeof(snap_buf),

                           s->name, sizeof(s->name),

                           conf, sizeof(conf)) < 0) {

        return -EINVAL;

    }



    clientname = qemu_rbd_parse_clientname(conf, clientname_buf);

    r = rados_create(&s->cluster, clientname);

    if (r < 0) {

        error_report("error initializing");

        return r;

    }



    s->snap = NULL;

    if (snap_buf[0] != '\0') {

        s->snap = g_strdup(snap_buf);

    }



    /*

     * Fallback to more conservative semantics if setting cache

     * options fails. Ignore errors from setting rbd_cache because the

     * only possible error is that the option does not exist, and

     * librbd defaults to no caching. If write through caching cannot

     * be set up, fall back to no caching.

     */

    if (flags & BDRV_O_NOCACHE) {

        rados_conf_set(s->cluster, "rbd_cache", "false");

    } else {

        rados_conf_set(s->cluster, "rbd_cache", "true");

    }



    if (strstr(conf, "conf=") == NULL) {

        /* try default location, but ignore failure */

        rados_conf_read_file(s->cluster, NULL);

    }



    if (conf[0] != '\0') {

        r = qemu_rbd_set_conf(s->cluster, conf);

        if (r < 0) {

            error_report("error setting config options");

            goto failed_shutdown;

        }

    }



    r = rados_connect(s->cluster);

    if (r < 0) {

        error_report("error connecting");

        goto failed_shutdown;

    }



    r = rados_ioctx_create(s->cluster, pool, &s->io_ctx);

    if (r < 0) {

        error_report("error opening pool %s", pool);

        goto failed_shutdown;

    }



    r = rbd_open(s->io_ctx, s->name, &s->image, s->snap);

    if (r < 0) {

        error_report("error reading header from %s", s->name);

        goto failed_open;

    }



    bs->read_only = (s->snap != NULL);



    s->event_reader_pos = 0;

    r = qemu_pipe(s->fds);

    if (r < 0) {

        error_report("error opening eventfd");

        goto failed;

    }

    fcntl(s->fds[0], F_SETFL, O_NONBLOCK);

    fcntl(s->fds[1], F_SETFL, O_NONBLOCK);

    qemu_aio_set_fd_handler(s->fds[RBD_FD_READ], qemu_rbd_aio_event_reader,

                            NULL, qemu_rbd_aio_flush_cb, s);





    return 0;



failed:

    rbd_close(s->image);

failed_open:

    rados_ioctx_destroy(s->io_ctx);

failed_shutdown:

    rados_shutdown(s->cluster);

    g_free(s->snap);

    return r;

}
