int ide_init_drive(IDEState *s, BlockDriverState *bs, IDEDriveKind kind,

                   const char *version, const char *serial, const char *model,

                   uint64_t wwn,

                   uint32_t cylinders, uint32_t heads, uint32_t secs,

                   int chs_trans)

{

    uint64_t nb_sectors;



    s->bs = bs;

    s->drive_kind = kind;



    bdrv_get_geometry(bs, &nb_sectors);

    if (cylinders < 1 || cylinders > 16383) {

        error_report("cyls must be between 1 and 16383");

        return -1;

    }

    if (heads < 1 || heads > 16) {

        error_report("heads must be between 1 and 16");

        return -1;

    }

    if (secs < 1 || secs > 63) {

        error_report("secs must be between 1 and 63");

        return -1;

    }

    s->cylinders = cylinders;

    s->heads = heads;

    s->sectors = secs;

    s->chs_trans = chs_trans;

    s->nb_sectors = nb_sectors;

    s->wwn = wwn;

    /* The SMART values should be preserved across power cycles

       but they aren't.  */

    s->smart_enabled = 1;

    s->smart_autosave = 1;

    s->smart_errors = 0;

    s->smart_selftest_count = 0;

    if (kind == IDE_CD) {

        bdrv_set_dev_ops(bs, &ide_cd_block_ops, s);

        bdrv_set_buffer_alignment(bs, 2048);

    } else {

        if (!bdrv_is_inserted(s->bs)) {

            error_report("Device needs media, but drive is empty");

            return -1;

        }

        if (bdrv_is_read_only(bs)) {

            error_report("Can't use a read-only drive");

            return -1;

        }

    }

    if (serial) {

        pstrcpy(s->drive_serial_str, sizeof(s->drive_serial_str), serial);

    } else {

        snprintf(s->drive_serial_str, sizeof(s->drive_serial_str),

                 "QM%05d", s->drive_serial);

    }

    if (model) {

        pstrcpy(s->drive_model_str, sizeof(s->drive_model_str), model);

    } else {

        switch (kind) {

        case IDE_CD:

            strcpy(s->drive_model_str, "QEMU DVD-ROM");

            break;

        case IDE_CFATA:

            strcpy(s->drive_model_str, "QEMU MICRODRIVE");

            break;

        default:

            strcpy(s->drive_model_str, "QEMU HARDDISK");

            break;

        }

    }



    if (version) {

        pstrcpy(s->version, sizeof(s->version), version);

    } else {

        pstrcpy(s->version, sizeof(s->version), qemu_get_version());

    }



    ide_reset(s);

    bdrv_iostatus_enable(bs);

    return 0;

}
