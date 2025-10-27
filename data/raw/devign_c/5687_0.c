static int qemu_gluster_open(BlockDriverState *bs,  QDict *options,

                             int bdrv_flags, Error **errp)

{

    BDRVGlusterState *s = bs->opaque;

    int open_flags = O_BINARY;

    int ret = 0;

    GlusterConf *gconf = g_malloc0(sizeof(GlusterConf));

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *filename;



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto out;

    }



    filename = qemu_opt_get(opts, "filename");



    s->glfs = qemu_gluster_init(gconf, filename, errp);

    if (!s->glfs) {

        ret = -errno;

        goto out;

    }



    if (bdrv_flags & BDRV_O_RDWR) {

        open_flags |= O_RDWR;

    } else {

        open_flags |= O_RDONLY;

    }



    if ((bdrv_flags & BDRV_O_NOCACHE)) {

        open_flags |= O_DIRECT;

    }



    s->fd = glfs_open(s->glfs, gconf->image, open_flags);

    if (!s->fd) {

        ret = -errno;

    }



out:

    qemu_opts_del(opts);

    qemu_gluster_gconf_free(gconf);

    if (!ret) {

        return ret;

    }

    if (s->fd) {

        glfs_close(s->fd);

    }

    if (s->glfs) {

        glfs_fini(s->glfs);

    }

    return ret;

}
