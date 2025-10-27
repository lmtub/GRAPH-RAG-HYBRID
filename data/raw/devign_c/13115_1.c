static int qemu_rbd_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVRBDState *s = bs->opaque;

    char pool[RBD_MAX_POOL_NAME_SIZE];

    char snap_buf[RBD_MAX_SNAP_NAME_SIZE];

    char conf[RBD_MAX_CONF_SIZE];

    char clientname_buf[RBD_MAX_CONF_SIZE];

    char *clientname;

    int r;



    if (qemu_rbd_parsename(filename, pool, sizeof(pool),

                           snap_buf, sizeof(snap_buf),

                           s->name, sizeof(s->name),

                           conf, sizeof(conf)) < 0) {

        return -EINVAL;

    }

    s->snap = NULL;

    if (snap_buf[0] != '\0') {

        s->snap = g_strdup(snap_buf);

    }



    clientname = qemu_rbd_parse_clientname(conf, clientname_buf);

    r = rados_create(&s->cluster, clientname);

    if (r < 0) {

        error_report("error initializing");

        return r;

    }



    if (strstr(conf, "conf=") == NULL) {

        r = rados_conf_read_file(s->cluster, NULL);

        if (r < 0) {

            error_report("error reading config file");

            rados_shutdown(s->cluster);

            return r;

        }

    }



    if (conf[0] != '\0') {

        r = qemu_rbd_set_conf(s->cluster, conf);

        if (r < 0) {

            error_report("error setting config options");

            rados_shutdown(s->cluster);

            return r;

        }

    }



    r = rados_connect(s->cluster);

    if (r < 0) {

        error_report("error connecting");

        rados_shutdown(s->cluster);

        return r;

    }



    r = rados_ioctx_create(s->cluster, pool, &s->io_ctx);

    if (r < 0) {

        error_report("error opening pool %s", pool);

        rados_shutdown(s->cluster);

        return r;

    }



    r = rbd_open(s->io_ctx, s->name, &s->image, s->snap);

    if (r < 0) {

        error_report("error reading header from %s", s->name);

        rados_ioctx_destroy(s->io_ctx);

        rados_shutdown(s->cluster);

        return r;

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

                            NULL, qemu_rbd_aio_flush_cb, NULL, s);





    return 0;



failed:

    rbd_close(s->image);

    rados_ioctx_destroy(s->io_ctx);

    rados_shutdown(s->cluster);

    return r;

}
