DriveInfo *drive_init(QemuOpts *opts, int default_to_scsi)

{

    const char *buf;

    const char *file = NULL;

    char devname[128];

    const char *serial;

    const char *mediastr = "";

    BlockInterfaceType type;

    enum { MEDIA_DISK, MEDIA_CDROM } media;

    int bus_id, unit_id;

    int cyls, heads, secs, translation;

    BlockDriver *drv = NULL;

    int max_devs;

    int index;

    int ro = 0;

    int bdrv_flags = 0;

    int on_read_error, on_write_error;

    const char *devaddr;

    DriveInfo *dinfo;

    BlockIOLimit io_limits;

    int snapshot = 0;

    bool copy_on_read;

    int ret;



    translation = BIOS_ATA_TRANSLATION_AUTO;

    media = MEDIA_DISK;



    /* extract parameters */

    bus_id  = qemu_opt_get_number(opts, "bus", 0);

    unit_id = qemu_opt_get_number(opts, "unit", -1);

    index   = qemu_opt_get_number(opts, "index", -1);



    cyls  = qemu_opt_get_number(opts, "cyls", 0);

    heads = qemu_opt_get_number(opts, "heads", 0);

    secs  = qemu_opt_get_number(opts, "secs", 0);



    snapshot = qemu_opt_get_bool(opts, "snapshot", 0);

    ro = qemu_opt_get_bool(opts, "readonly", 0);

    copy_on_read = qemu_opt_get_bool(opts, "copy-on-read", false);



    file = qemu_opt_get(opts, "file");

    serial = qemu_opt_get(opts, "serial");



    if ((buf = qemu_opt_get(opts, "if")) != NULL) {

        pstrcpy(devname, sizeof(devname), buf);

        for (type = 0; type < IF_COUNT && strcmp(buf, if_name[type]); type++)

            ;

        if (type == IF_COUNT) {

            error_report("unsupported bus type '%s'", buf);

            return NULL;

	}

    } else {

        type = default_to_scsi ? IF_SCSI : IF_IDE;

        pstrcpy(devname, sizeof(devname), if_name[type]);

    }



    max_devs = if_max_devs[type];



    if (cyls || heads || secs) {

        if (cyls < 1 || (type == IF_IDE && cyls > 16383)) {

            error_report("invalid physical cyls number");

	    return NULL;

	}

        if (heads < 1 || (type == IF_IDE && heads > 16)) {

            error_report("invalid physical heads number");

	    return NULL;

	}

        if (secs < 1 || (type == IF_IDE && secs > 63)) {

            error_report("invalid physical secs number");

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "trans")) != NULL) {

        if (!cyls) {

            error_report("'%s' trans must be used with cyls, heads and secs",

                         buf);

            return NULL;

        }

        if (!strcmp(buf, "none"))

            translation = BIOS_ATA_TRANSLATION_NONE;

        else if (!strcmp(buf, "lba"))

            translation = BIOS_ATA_TRANSLATION_LBA;

        else if (!strcmp(buf, "auto"))

            translation = BIOS_ATA_TRANSLATION_AUTO;

	else {

            error_report("'%s' invalid translation type", buf);

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "media")) != NULL) {

        if (!strcmp(buf, "disk")) {

	    media = MEDIA_DISK;

	} else if (!strcmp(buf, "cdrom")) {

            if (cyls || secs || heads) {

                error_report("CHS can't be set with media=%s", buf);

	        return NULL;

            }

	    media = MEDIA_CDROM;

	} else {

	    error_report("'%s' invalid media", buf);

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "cache")) != NULL) {

        if (bdrv_parse_cache_flags(buf, &bdrv_flags) != 0) {

            error_report("invalid cache option");

            return NULL;

        }

    }



#ifdef CONFIG_LINUX_AIO

    if ((buf = qemu_opt_get(opts, "aio")) != NULL) {

        if (!strcmp(buf, "native")) {

            bdrv_flags |= BDRV_O_NATIVE_AIO;

        } else if (!strcmp(buf, "threads")) {

            /* this is the default */

        } else {

           error_report("invalid aio option");

           return NULL;

        }

    }

#endif



    if ((buf = qemu_opt_get(opts, "format")) != NULL) {

       if (strcmp(buf, "?") == 0) {

           error_printf("Supported formats:");

           bdrv_iterate_format(bdrv_format_print, NULL);

           error_printf("\n");

           return NULL;

        }

        drv = bdrv_find_whitelisted_format(buf);

        if (!drv) {

            error_report("'%s' invalid format", buf);

            return NULL;

        }

    }



    /* disk I/O throttling */

    io_limits.bps[BLOCK_IO_LIMIT_TOTAL]  =

                           qemu_opt_get_number(opts, "bps", 0);

    io_limits.bps[BLOCK_IO_LIMIT_READ]   =

                           qemu_opt_get_number(opts, "bps_rd", 0);

    io_limits.bps[BLOCK_IO_LIMIT_WRITE]  =

                           qemu_opt_get_number(opts, "bps_wr", 0);

    io_limits.iops[BLOCK_IO_LIMIT_TOTAL] =

                           qemu_opt_get_number(opts, "iops", 0);

    io_limits.iops[BLOCK_IO_LIMIT_READ]  =

                           qemu_opt_get_number(opts, "iops_rd", 0);

    io_limits.iops[BLOCK_IO_LIMIT_WRITE] =

                           qemu_opt_get_number(opts, "iops_wr", 0);



    if (!do_check_io_limits(&io_limits)) {

        error_report("bps(iops) and bps_rd/bps_wr(iops_rd/iops_wr) "

                     "cannot be used at the same time");

        return NULL;

    }



    on_write_error = BLOCK_ERR_STOP_ENOSPC;

    if ((buf = qemu_opt_get(opts, "werror")) != NULL) {

        if (type != IF_IDE && type != IF_SCSI && type != IF_VIRTIO && type != IF_NONE) {

            error_report("werror is not supported by this bus type");

            return NULL;

        }



        on_write_error = parse_block_error_action(buf, 0);

        if (on_write_error < 0) {

            return NULL;

        }

    }



    on_read_error = BLOCK_ERR_REPORT;

    if ((buf = qemu_opt_get(opts, "rerror")) != NULL) {

        if (type != IF_IDE && type != IF_VIRTIO && type != IF_SCSI && type != IF_NONE) {

            error_report("rerror is not supported by this bus type");

            return NULL;

        }



        on_read_error = parse_block_error_action(buf, 1);

        if (on_read_error < 0) {

            return NULL;

        }

    }



    if ((devaddr = qemu_opt_get(opts, "addr")) != NULL) {

        if (type != IF_VIRTIO) {

            error_report("addr is not supported by this bus type");

            return NULL;

        }

    }



    /* compute bus and unit according index */



    if (index != -1) {

        if (bus_id != 0 || unit_id != -1) {

            error_report("index cannot be used with bus and unit");

            return NULL;

        }

        bus_id = drive_index_to_bus_id(type, index);

        unit_id = drive_index_to_unit_id(type, index);

    }



    /* if user doesn't specify a unit_id,

     * try to find the first free

     */



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



    /* check unit id */



    if (max_devs && unit_id >= max_devs) {

        error_report("unit %d too big (max is %d)",

                     unit_id, max_devs - 1);

        return NULL;

    }



    /*

     * catch multiple definitions

     */



    if (drive_get(type, bus_id, unit_id) != NULL) {

        error_report("drive with bus=%d, unit=%d (index=%d) exists",

                     bus_id, unit_id, index);

        return NULL;

    }



    /* init */



    dinfo = g_malloc0(sizeof(*dinfo));

    if ((buf = qemu_opts_id(opts)) != NULL) {

        dinfo->id = g_strdup(buf);

    } else {

        /* no id supplied -> create one */

        dinfo->id = g_malloc0(32);

        if (type == IF_IDE || type == IF_SCSI)

            mediastr = (media == MEDIA_CDROM) ? "-cd" : "-hd";

        if (max_devs)

            snprintf(dinfo->id, 32, "%s%i%s%i",

                     devname, bus_id, mediastr, unit_id);

        else

            snprintf(dinfo->id, 32, "%s%s%i",

                     devname, mediastr, unit_id);

    }

    dinfo->bdrv = bdrv_new(dinfo->id);

    dinfo->devaddr = devaddr;

    dinfo->type = type;

    dinfo->bus = bus_id;

    dinfo->unit = unit_id;

    dinfo->opts = opts;

    dinfo->refcount = 1;

    if (serial) {

        pstrcpy(dinfo->serial, sizeof(dinfo->serial), serial);

    }

    QTAILQ_INSERT_TAIL(&drives, dinfo, next);



    bdrv_set_on_error(dinfo->bdrv, on_read_error, on_write_error);



    /* disk I/O throttling */

    bdrv_set_io_limits(dinfo->bdrv, &io_limits);



    switch(type) {

    case IF_IDE:

    case IF_SCSI:

    case IF_XEN:

    case IF_NONE:

        switch(media) {

	case MEDIA_DISK:

            if (cyls != 0) {

                bdrv_set_geometry_hint(dinfo->bdrv, cyls, heads, secs);

                bdrv_set_translation_hint(dinfo->bdrv, translation);

            }

	    break;

	case MEDIA_CDROM:

            dinfo->media_cd = 1;

	    break;

	}

        break;

    case IF_SD:

    case IF_FLOPPY:

    case IF_PFLASH:

    case IF_MTD:

        break;

    case IF_VIRTIO:

        /* add virtio block device */

        opts = qemu_opts_create(qemu_find_opts("device"), NULL, 0);

        if (arch_type == QEMU_ARCH_S390X) {

            qemu_opt_set(opts, "driver", "virtio-blk-s390");

        } else {

            qemu_opt_set(opts, "driver", "virtio-blk-pci");

        }

        qemu_opt_set(opts, "drive", dinfo->id);

        if (devaddr)

            qemu_opt_set(opts, "addr", devaddr);

        break;

    default:

        abort();

    }

    if (!file || !*file) {

        return dinfo;

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



    if (media == MEDIA_CDROM) {

        /* CDROM is fine for any interface, don't check.  */

        ro = 1;

    } else if (ro == 1) {

        if (type != IF_SCSI && type != IF_VIRTIO && type != IF_FLOPPY &&

            type != IF_NONE && type != IF_PFLASH) {

            error_report("readonly not supported by this bus type");

            goto err;

        }

    }



    bdrv_flags |= ro ? 0 : BDRV_O_RDWR;



    ret = bdrv_open(dinfo->bdrv, file, bdrv_flags, drv);

    if (ret < 0) {

        error_report("could not open disk image %s: %s",

                     file, strerror(-ret));

        goto err;

    }



    if (bdrv_key_required(dinfo->bdrv))

        autostart = 0;

    return dinfo;



err:

    bdrv_delete(dinfo->bdrv);

    g_free(dinfo->id);

    QTAILQ_REMOVE(&drives, dinfo, next);

    g_free(dinfo);

    return NULL;

}
