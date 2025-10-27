static int enable_write_target(BlockDriverState *bs, Error **errp)

{

    BDRVVVFATState *s = bs->opaque;

    BlockDriver *bdrv_qcow = NULL;

    BlockDriverState *backing;

    QemuOpts *opts = NULL;

    int ret;

    int size = sector2cluster(s, s->sector_count);

    QDict *options;



    s->used_clusters = calloc(size, 1);



    array_init(&(s->commits), sizeof(commit_t));



    s->qcow_filename = g_malloc(PATH_MAX);

    ret = get_tmp_filename(s->qcow_filename, PATH_MAX);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "can't create temporary file");

        goto err;

    }



    bdrv_qcow = bdrv_find_format("qcow");

    if (!bdrv_qcow) {

        error_setg(errp, "Failed to locate qcow driver");

        ret = -ENOENT;

        goto err;

    }



    opts = qemu_opts_create(bdrv_qcow->create_opts, NULL, 0, &error_abort);

    qemu_opt_set_number(opts, BLOCK_OPT_SIZE, s->sector_count * 512,

                        &error_abort);

    qemu_opt_set(opts, BLOCK_OPT_BACKING_FILE, "fat:", &error_abort);



    ret = bdrv_create(bdrv_qcow, s->qcow_filename, opts, errp);

    qemu_opts_del(opts);

    if (ret < 0) {

        goto err;

    }



    options = qdict_new();

    qdict_put(options, "write-target.driver", qstring_from_str("qcow"));

    s->qcow = bdrv_open_child(s->qcow_filename, options, "write-target", bs,

                              &child_vvfat_qcow, false, errp);

    QDECREF(options);

    if (!s->qcow) {

        ret = -EINVAL;

        goto err;

    }



#ifndef _WIN32

    unlink(s->qcow_filename);

#endif



    backing = bdrv_new();

    bdrv_set_backing_hd(s->bs, backing);

    bdrv_unref(backing);



    s->bs->backing->bs->drv = &vvfat_write_target;

    s->bs->backing->bs->opaque = g_new(void *, 1);

    *(void**)s->bs->backing->bs->opaque = s;



    return 0;



err:

    g_free(s->qcow_filename);

    s->qcow_filename = NULL;

    return ret;

}
