static int qemu_rbd_open(BlockDriverState *bs, QDict *options, int flags,

                         Error **errp)

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



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        qemu_opts_del(opts);

        return -EINVAL;

    }



    filename = qemu_opt_get(opts, "filename");



    if (qemu_rbd_parsename(filename, pool, sizeof(pool),

                           snap_buf, sizeof(snap_buf),

                           s->name, sizeof(s->name),

                           conf, sizeof(conf), errp) < 0) {

        r = -EINVAL;

        goto failed_opts;

    }



    clientname = qemu_rbd_parse_clientname(conf, clientname_buf);

    r = rados_create(&s->cluster, clientname);

    if (r < 0) {

        error_setg(&local_err, "error initializing");

        goto failed_opts;

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

        r = qemu_rbd_set_conf(s->cluster, conf, errp);

        if (r < 0) {

            goto failed_shutdown;

        }

    }



    r = rados_connect(s->cluster);

    if (r < 0) {

        error_setg(&local_err, "error connecting");

        goto failed_shutdown;

    }



    r = rados_ioctx_create(s->cluster, pool, &s->io_ctx);

    if (r < 0) {

        error_setg(&local_err, "error opening pool %s", pool);

        goto failed_shutdown;

    }



    r = rbd_open(s->io_ctx, s->name, &s->image, s->snap);

    if (r < 0) {

        error_setg(&local_err, "error reading header from %s", s->name);

        goto failed_open;

    }



    bs->read_only = (s->snap != NULL);



    qemu_opts_del(opts);

    return 0;



failed_open:

    rados_ioctx_destroy(s->io_ctx);

failed_shutdown:

    rados_shutdown(s->cluster);

    g_free(s->snap);

failed_opts:

    qemu_opts_del(opts);

    return r;

}
