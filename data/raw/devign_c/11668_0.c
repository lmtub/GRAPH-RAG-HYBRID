static int qemu_rbd_create(const char *filename, QemuOpts *opts, Error **errp)

{

    Error *local_err = NULL;

    int64_t bytes = 0;

    int64_t objsize;

    int obj_order = 0;

    const char *pool, *name, *conf, *clientname, *keypairs;

    const char *secretid;

    rados_t cluster;

    rados_ioctx_t io_ctx;

    QDict *options = NULL;

    QemuOpts *rbd_opts = NULL;

    int ret = 0;



    secretid = qemu_opt_get(opts, "password-secret");



    /* Read out options */

    bytes = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                     BDRV_SECTOR_SIZE);

    objsize = qemu_opt_get_size_del(opts, BLOCK_OPT_CLUSTER_SIZE, 0);

    if (objsize) {

        if ((objsize - 1) & objsize) {    /* not a power of 2? */

            error_setg(errp, "obj size needs to be power of 2");

            ret = -EINVAL;

            goto exit;

        }

        if (objsize < 4096) {

            error_setg(errp, "obj size too small");

            ret = -EINVAL;

            goto exit;

        }

        obj_order = ctz32(objsize);

    }



    options = qdict_new();

    qemu_rbd_parse_filename(filename, options, &local_err);

    if (local_err) {

        ret = -EINVAL;

        error_propagate(errp, local_err);

        goto exit;

    }



    rbd_opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(rbd_opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto exit;

    }



    pool       = qemu_opt_get(rbd_opts, "pool");

    conf       = qemu_opt_get(rbd_opts, "conf");

    clientname = qemu_opt_get(rbd_opts, "user");

    name       = qemu_opt_get(rbd_opts, "image");

    keypairs   = qemu_opt_get(rbd_opts, "keyvalue-pairs");



    ret = rados_create(&cluster, clientname);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "error initializing");

        goto exit;

    }



    /* try default location when conf=NULL, but ignore failure */

    ret = rados_conf_read_file(cluster, conf);

    if (conf && ret < 0) {

        error_setg_errno(errp, -ret, "error reading conf file %s", conf);

        ret = -EIO;

        goto shutdown;

    }



    ret = qemu_rbd_set_keypairs(cluster, keypairs, errp);

    if (ret < 0) {

        ret = -EIO;

        goto shutdown;

    }



    if (qemu_rbd_set_auth(cluster, secretid, errp) < 0) {

        ret = -EIO;

        goto shutdown;

    }



    ret = rados_connect(cluster);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "error connecting");

        goto shutdown;

    }



    ret = rados_ioctx_create(cluster, pool, &io_ctx);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "error opening pool %s", pool);

        goto shutdown;

    }



    ret = rbd_create(io_ctx, name, bytes, &obj_order);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "error rbd create");

    }



    rados_ioctx_destroy(io_ctx);



shutdown:

    rados_shutdown(cluster);



exit:

    QDECREF(options);

    qemu_opts_del(rbd_opts);

    return ret;

}
