DriveInfo *drive_new(QemuOpts *all_opts, BlockInterfaceType block_default_type)

{

    const char *value;

    BlockBackend *blk;

    DriveInfo *dinfo = NULL;

    QDict *bs_opts;

    QemuOpts *legacy_opts;

    DriveMediaType media = MEDIA_DISK;

    BlockInterfaceType type;

    int cyls, heads, secs, translation;

    int max_devs, bus_id, unit_id, index;

    const char *devaddr;

    const char *werror, *rerror;

    bool read_only = false;

    bool copy_on_read;

    const char *serial;

    const char *filename;

    Error *local_err = NULL;

    int i;

    const char *deprecated[] = {

        "serial", "trans", "secs", "heads", "cyls", "addr"

    };



    /* Change legacy command line options into QMP ones */

    static const struct {

        const char *from;

        const char *to;

    } opt_renames[] = {

        { "iops",           "throttling.iops-total" },

        { "iops_rd",        "throttling.iops-read" },

        { "iops_wr",        "throttling.iops-write" },



        { "bps",            "throttling.bps-total" },

        { "bps_rd",         "throttling.bps-read" },

        { "bps_wr",         "throttling.bps-write" },



        { "iops_max",       "throttling.iops-total-max" },

        { "iops_rd_max",    "throttling.iops-read-max" },

        { "iops_wr_max",    "throttling.iops-write-max" },



        { "bps_max",        "throttling.bps-total-max" },

        { "bps_rd_max",     "throttling.bps-read-max" },

        { "bps_wr_max",     "throttling.bps-write-max" },



        { "iops_size",      "throttling.iops-size" },



        { "group",          "throttling.group" },



        { "readonly",       BDRV_OPT_READ_ONLY },

    };



    for (i = 0; i < ARRAY_SIZE(opt_renames); i++) {

        qemu_opt_rename(all_opts, opt_renames[i].from, opt_renames[i].to,

                        &local_err);

        if (local_err) {

            error_report_err(local_err);

            return NULL;

        }

    }



    value = qemu_opt_get(all_opts, "cache");

    if (value) {

        int flags = 0;

        bool writethrough;



        if (bdrv_parse_cache_mode(value, &flags, &writethrough) != 0) {

            error_report("invalid cache option");

            return NULL;

        }



        /* Specific options take precedence */

        if (!qemu_opt_get(all_opts, BDRV_OPT_CACHE_WB)) {

            qemu_opt_set_bool(all_opts, BDRV_OPT_CACHE_WB,

                              !writethrough, &error_abort);

        }

        if (!qemu_opt_get(all_opts, BDRV_OPT_CACHE_DIRECT)) {

            qemu_opt_set_bool(all_opts, BDRV_OPT_CACHE_DIRECT,

                              !!(flags & BDRV_O_NOCACHE), &error_abort);

        }

        if (!qemu_opt_get(all_opts, BDRV_OPT_CACHE_NO_FLUSH)) {

            qemu_opt_set_bool(all_opts, BDRV_OPT_CACHE_NO_FLUSH,

                              !!(flags & BDRV_O_NO_FLUSH), &error_abort);

        }

        qemu_opt_unset(all_opts, "cache");

    }



    /* Get a QDict for processing the options */

    bs_opts = qdict_new();

    qemu_opts_to_qdict(all_opts, bs_opts);



    legacy_opts = qemu_opts_create(&qemu_legacy_drive_opts, NULL, 0,

                                   &error_abort);

    qemu_opts_absorb_qdict(legacy_opts, bs_opts, &local_err);

    if (local_err) {

        error_report_err(local_err);

        goto fail;

    }



    /* Deprecated option boot=[on|off] */

    if (qemu_opt_get(legacy_opts, "boot") != NULL) {

        fprintf(stderr, "qemu-kvm: boot=on|off is deprecated and will be "

                "ignored. Future versions will reject this parameter. Please "

                "update your scripts.\n");

    }



    /* Other deprecated options */

    if (!qtest_enabled()) {

        for (i = 0; i < ARRAY_SIZE(deprecated); i++) {

            if (qemu_opt_get(legacy_opts, deprecated[i]) != NULL) {

                error_report("'%s' is deprecated, please use the corresponding "

                             "option of '-device' instead", deprecated[i]);

            }

        }

    }



    /* Media type */

    value = qemu_opt_get(legacy_opts, "media");

    if (value) {

        if (!strcmp(value, "disk")) {

            media = MEDIA_DISK;

        } else if (!strcmp(value, "cdrom")) {

            media = MEDIA_CDROM;

            read_only = true;

        } else {

            error_report("'%s' invalid media", value);

            goto fail;

        }

    }



    /* copy-on-read is disabled with a warning for read-only devices */

    read_only |= qemu_opt_get_bool(legacy_opts, BDRV_OPT_READ_ONLY, false);

    copy_on_read = qemu_opt_get_bool(legacy_opts, "copy-on-read", false);



    if (read_only && copy_on_read) {

        error_report("warning: disabling copy-on-read on read-only drive");

        copy_on_read = false;

    }



    qdict_put_str(bs_opts, BDRV_OPT_READ_ONLY, read_only ? "on" : "off");

    qdict_put_str(bs_opts, "copy-on-read", copy_on_read ? "on" : "off");



    /* Controller type */

    value = qemu_opt_get(legacy_opts, "if");

    if (value) {

        for (type = 0;

             type < IF_COUNT && strcmp(value, if_name[type]);

             type++) {

        }

        if (type == IF_COUNT) {

            error_report("unsupported bus type '%s'", value);

            goto fail;

        }

    } else {

        type = block_default_type;

    }



    /* Geometry */

    cyls  = qemu_opt_get_number(legacy_opts, "cyls", 0);

    heads = qemu_opt_get_number(legacy_opts, "heads", 0);

    secs  = qemu_opt_get_number(legacy_opts, "secs", 0);



    if (cyls || heads || secs) {

        if (cyls < 1) {

            error_report("invalid physical cyls number");

            goto fail;

        }

        if (heads < 1) {

            error_report("invalid physical heads number");

            goto fail;

        }

        if (secs < 1) {

            error_report("invalid physical secs number");

            goto fail;

        }

    }



    translation = BIOS_ATA_TRANSLATION_AUTO;

    value = qemu_opt_get(legacy_opts, "trans");

    if (value != NULL) {

        if (!cyls) {

            error_report("'%s' trans must be used with cyls, heads and secs",

                         value);

            goto fail;

        }

        if (!strcmp(value, "none")) {

            translation = BIOS_ATA_TRANSLATION_NONE;

        } else if (!strcmp(value, "lba")) {

            translation = BIOS_ATA_TRANSLATION_LBA;

        } else if (!strcmp(value, "large")) {

            translation = BIOS_ATA_TRANSLATION_LARGE;

        } else if (!strcmp(value, "rechs")) {

            translation = BIOS_ATA_TRANSLATION_RECHS;

        } else if (!strcmp(value, "auto")) {

            translation = BIOS_ATA_TRANSLATION_AUTO;

        } else {

            error_report("'%s' invalid translation type", value);

            goto fail;

        }

    }



    if (media == MEDIA_CDROM) {

        if (cyls || secs || heads) {

            error_report("CHS can't be set with media=cdrom");

            goto fail;

        }

    }



    /* Device address specified by bus/unit or index.

     * If none was specified, try to find the first free one. */

    bus_id  = qemu_opt_get_number(legacy_opts, "bus", 0);

    unit_id = qemu_opt_get_number(legacy_opts, "unit", -1);

    index   = qemu_opt_get_number(legacy_opts, "index", -1);



    max_devs = if_max_devs[type];



    if (index != -1) {

        if (bus_id != 0 || unit_id != -1) {

            error_report("index cannot be used with bus and unit");

            goto fail;

        }

        bus_id = drive_index_to_bus_id(type, index);

        unit_id = drive_index_to_unit_id(type, index);

    }



    if (unit_id == -1) {

       unit_id = 0;

       while (drive_get(type, bus_id, unit_id) != NULL) {

           unit_id++;

           if (max_devs && unit_id >= max_devs) {

               unit_id -= max_devs;

               bus_id++;

           }

       }

    }



    if (max_devs && unit_id >= max_devs) {

        error_report("unit %d too big (max is %d)", unit_id, max_devs - 1);

        goto fail;

    }



    if (drive_get(type, bus_id, unit_id) != NULL) {

        error_report("drive with bus=%d, unit=%d (index=%d) exists",

                     bus_id, unit_id, index);

        goto fail;

    }



    /* Serial number */

    serial = qemu_opt_get(legacy_opts, "serial");



    /* no id supplied -> create one */

    if (qemu_opts_id(all_opts) == NULL) {

        char *new_id;

        const char *mediastr = "";

        if (type == IF_IDE || type == IF_SCSI) {

            mediastr = (media == MEDIA_CDROM) ? "-cd" : "-hd";

        }

        if (max_devs) {

            new_id = g_strdup_printf("%s%i%s%i", if_name[type], bus_id,

                                     mediastr, unit_id);

        } else {

            new_id = g_strdup_printf("%s%s%i", if_name[type],

                                     mediastr, unit_id);

        }

        qdict_put_str(bs_opts, "id", new_id);

        g_free(new_id);

    }



    /* Add virtio block device */

    devaddr = qemu_opt_get(legacy_opts, "addr");

    if (devaddr && type != IF_VIRTIO) {

        error_report("addr is not supported by this bus type");

        goto fail;

    }



    if (type == IF_VIRTIO) {

        QemuOpts *devopts;

        devopts = qemu_opts_create(qemu_find_opts("device"), NULL, 0,

                                   &error_abort);

        if (arch_type == QEMU_ARCH_S390X) {

            qemu_opt_set(devopts, "driver", "virtio-blk-ccw", &error_abort);

        } else {

            qemu_opt_set(devopts, "driver", "virtio-blk-pci", &error_abort);

        }

        qemu_opt_set(devopts, "drive", qdict_get_str(bs_opts, "id"),

                     &error_abort);

        if (devaddr) {

            qemu_opt_set(devopts, "addr", devaddr, &error_abort);

        }

    }



    filename = qemu_opt_get(legacy_opts, "file");



    /* Check werror/rerror compatibility with if=... */

    werror = qemu_opt_get(legacy_opts, "werror");

    if (werror != NULL) {

        if (type != IF_IDE && type != IF_SCSI && type != IF_VIRTIO &&

            type != IF_NONE) {

            error_report("werror is not supported by this bus type");

            goto fail;

        }

        qdict_put_str(bs_opts, "werror", werror);

    }



    rerror = qemu_opt_get(legacy_opts, "rerror");

    if (rerror != NULL) {

        if (type != IF_IDE && type != IF_VIRTIO && type != IF_SCSI &&

            type != IF_NONE) {

            error_report("rerror is not supported by this bus type");

            goto fail;

        }

        qdict_put_str(bs_opts, "rerror", rerror);

    }



    /* Actual block device init: Functionality shared with blockdev-add */

    blk = blockdev_init(filename, bs_opts, &local_err);

    bs_opts = NULL;

    if (!blk) {

        if (local_err) {

            error_report_err(local_err);

        }

        goto fail;

    } else {

        assert(!local_err);

    }



    /* Create legacy DriveInfo */

    dinfo = g_malloc0(sizeof(*dinfo));

    dinfo->opts = all_opts;



    dinfo->cyls = cyls;

    dinfo->heads = heads;

    dinfo->secs = secs;

    dinfo->trans = translation;



    dinfo->type = type;

    dinfo->bus = bus_id;

    dinfo->unit = unit_id;

    dinfo->devaddr = devaddr;

    dinfo->serial = g_strdup(serial);



    blk_set_legacy_dinfo(blk, dinfo);



    switch(type) {

    case IF_IDE:

    case IF_SCSI:

    case IF_XEN:

    case IF_NONE:

        dinfo->media_cd = media == MEDIA_CDROM;

        break;

    default:

        break;

    }



fail:

    qemu_opts_del(legacy_opts);

    QDECREF(bs_opts);

    return dinfo;

}
