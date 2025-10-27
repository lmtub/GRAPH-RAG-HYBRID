static DriveInfo *blockdev_init(const char *file, QDict *bs_opts,

                                BlockInterfaceType type,

                                Error **errp)

{

    const char *buf;

    const char *serial;

    int ro = 0;

    int bdrv_flags = 0;

    int on_read_error, on_write_error;

    DriveInfo *dinfo;

    ThrottleConfig cfg;

    int snapshot = 0;

    bool copy_on_read;

    int ret;

    Error *error = NULL;

    QemuOpts *opts;

    const char *id;

    bool has_driver_specific_opts;

    BlockDriver *drv = NULL;



    /* Check common options by copying from bs_opts to opts, all other options

     * stay in bs_opts for processing by bdrv_open(). */

    id = qdict_get_try_str(bs_opts, "id");

    opts = qemu_opts_create(&qemu_common_drive_opts, id, 1, &error);

    if (error_is_set(&error)) {

        error_propagate(errp, error);

        return NULL;

    }



    qemu_opts_absorb_qdict(opts, bs_opts, &error);

    if (error_is_set(&error)) {

        error_propagate(errp, error);

        goto early_err;

    }



    if (id) {

        qdict_del(bs_opts, "id");

    }



    has_driver_specific_opts = !!qdict_size(bs_opts);



    /* extract parameters */

    snapshot = qemu_opt_get_bool(opts, "snapshot", 0);

    ro = qemu_opt_get_bool(opts, "read-only", 0);

    copy_on_read = qemu_opt_get_bool(opts, "copy-on-read", false);



    serial = qemu_opt_get(opts, "serial");



    if ((buf = qemu_opt_get(opts, "discard")) != NULL) {

        if (bdrv_parse_discard_flags(buf, &bdrv_flags) != 0) {

            error_setg(errp, "invalid discard option");

            goto early_err;

        }

    }



    if (qemu_opt_get_bool(opts, "cache.writeback", true)) {

        bdrv_flags |= BDRV_O_CACHE_WB;

    }

    if (qemu_opt_get_bool(opts, "cache.direct", false)) {

        bdrv_flags |= BDRV_O_NOCACHE;

    }

    if (qemu_opt_get_bool(opts, "cache.no-flush", false)) {

        bdrv_flags |= BDRV_O_NO_FLUSH;

    }



#ifdef CONFIG_LINUX_AIO

    if ((buf = qemu_opt_get(opts, "aio")) != NULL) {

        if (!strcmp(buf, "native")) {

            bdrv_flags |= BDRV_O_NATIVE_AIO;

        } else if (!strcmp(buf, "threads")) {

            /* this is the default */

        } else {

           error_setg(errp, "invalid aio option");

           goto early_err;

        }

    }

#endif



    if ((buf = qemu_opt_get(opts, "format")) != NULL) {

        if (is_help_option(buf)) {

            error_printf("Supported formats:");

            bdrv_iterate_format(bdrv_format_print, NULL);

            error_printf("\n");

            goto early_err;

        }



        drv = bdrv_find_format(buf);

        if (!drv) {

            error_setg(errp, "'%s' invalid format", buf);

            goto early_err;

        }

    }



    /* disk I/O throttling */

    memset(&cfg, 0, sizeof(cfg));

    cfg.buckets[THROTTLE_BPS_TOTAL].avg =

        qemu_opt_get_number(opts, "throttling.bps-total", 0);

    cfg.buckets[THROTTLE_BPS_READ].avg  =

        qemu_opt_get_number(opts, "throttling.bps-read", 0);

    cfg.buckets[THROTTLE_BPS_WRITE].avg =

        qemu_opt_get_number(opts, "throttling.bps-write", 0);

    cfg.buckets[THROTTLE_OPS_TOTAL].avg =

        qemu_opt_get_number(opts, "throttling.iops-total", 0);

    cfg.buckets[THROTTLE_OPS_READ].avg =

        qemu_opt_get_number(opts, "throttling.iops-read", 0);

    cfg.buckets[THROTTLE_OPS_WRITE].avg =

        qemu_opt_get_number(opts, "throttling.iops-write", 0);



    cfg.buckets[THROTTLE_BPS_TOTAL].max =

        qemu_opt_get_number(opts, "throttling.bps-total-max", 0);

    cfg.buckets[THROTTLE_BPS_READ].max  =

        qemu_opt_get_number(opts, "throttling.bps-read-max", 0);

    cfg.buckets[THROTTLE_BPS_WRITE].max =

        qemu_opt_get_number(opts, "throttling.bps-write-max", 0);

    cfg.buckets[THROTTLE_OPS_TOTAL].max =

        qemu_opt_get_number(opts, "throttling.iops-total-max", 0);

    cfg.buckets[THROTTLE_OPS_READ].max =

        qemu_opt_get_number(opts, "throttling.iops-read-max", 0);

    cfg.buckets[THROTTLE_OPS_WRITE].max =

        qemu_opt_get_number(opts, "throttling.iops-write-max", 0);



    cfg.op_size = qemu_opt_get_number(opts, "throttling.iops-size", 0);



    if (!check_throttle_config(&cfg, &error)) {

        error_propagate(errp, error);

        goto early_err;

    }



    on_write_error = BLOCKDEV_ON_ERROR_ENOSPC;

    if ((buf = qemu_opt_get(opts, "werror")) != NULL) {

        if (type != IF_IDE && type != IF_SCSI && type != IF_VIRTIO && type != IF_NONE) {

            error_setg(errp, "werror is not supported by this bus type");

            goto early_err;

        }



        on_write_error = parse_block_error_action(buf, 0, &error);

        if (error_is_set(&error)) {

            error_propagate(errp, error);

            goto early_err;

        }

    }



    on_read_error = BLOCKDEV_ON_ERROR_REPORT;

    if ((buf = qemu_opt_get(opts, "rerror")) != NULL) {

        if (type != IF_IDE && type != IF_VIRTIO && type != IF_SCSI && type != IF_NONE) {

            error_report("rerror is not supported by this bus type");

            goto early_err;

        }



        on_read_error = parse_block_error_action(buf, 1, &error);

        if (error_is_set(&error)) {

            error_propagate(errp, error);

            goto early_err;

        }

    }



    /* init */

    dinfo = g_malloc0(sizeof(*dinfo));

    dinfo->id = g_strdup(qemu_opts_id(opts));

    dinfo->bdrv = bdrv_new(dinfo->id);

    dinfo->bdrv->open_flags = snapshot ? BDRV_O_SNAPSHOT : 0;

    dinfo->bdrv->read_only = ro;

    dinfo->type = type;

    dinfo->refcount = 1;

    if (serial != NULL) {

        dinfo->serial = g_strdup(serial);

    }

    QTAILQ_INSERT_TAIL(&drives, dinfo, next);



    bdrv_set_on_error(dinfo->bdrv, on_read_error, on_write_error);



    /* disk I/O throttling */

    if (throttle_enabled(&cfg)) {

        bdrv_io_limits_enable(dinfo->bdrv);

        bdrv_set_io_limits(dinfo->bdrv, &cfg);

    }



    if (!file || !*file) {

        if (has_driver_specific_opts) {

            file = NULL;

        } else {

            QDECREF(bs_opts);

            qemu_opts_del(opts);

            return dinfo;

        }

    }

    if (snapshot) {

        /* always use cache=unsafe with snapshot */

        bdrv_flags &= ~BDRV_O_CACHE_MASK;

        bdrv_flags |= (BDRV_O_SNAPSHOT|BDRV_O_CACHE_WB|BDRV_O_NO_FLUSH);

    }



    if (copy_on_read) {

        bdrv_flags |= BDRV_O_COPY_ON_READ;

    }



    if (runstate_check(RUN_STATE_INMIGRATE)) {

        bdrv_flags |= BDRV_O_INCOMING;

    }



    bdrv_flags |= ro ? 0 : BDRV_O_RDWR;



    QINCREF(bs_opts);

    ret = bdrv_open(dinfo->bdrv, file, bs_opts, bdrv_flags, drv, &error);



    if (ret < 0) {

        error_setg(errp, "could not open disk image %s: %s",

                   file ?: dinfo->id, error_get_pretty(error));

        error_free(error);

        goto err;

    }



    if (bdrv_key_required(dinfo->bdrv))

        autostart = 0;



    QDECREF(bs_opts);

    qemu_opts_del(opts);



    return dinfo;



err:

    bdrv_unref(dinfo->bdrv);

    g_free(dinfo->id);

    QTAILQ_REMOVE(&drives, dinfo, next);

    g_free(dinfo);

early_err:

    QDECREF(bs_opts);

    qemu_opts_del(opts);

    return NULL;

}
