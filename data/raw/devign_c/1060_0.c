int ide_init_drive(IDEState *s, BlockBackend *blk, IDEDriveKind kind,

                   const char *version, const char *serial, const char *model,

                   uint64_t wwn,

                   uint32_t cylinders, uint32_t heads, uint32_t secs,

                   int chs_trans)

{

    uint64_t nb_sectors;



    s->blk = blk;

    s->drive_kind = kind;



    blk_get_geometry(blk, &nb_sectors);

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

        blk_set_dev_ops(blk, &ide_cd_block_ops, s);

        blk_set_guest_block_size(blk, 2048);

    } else {

        if (!blk_is_inserted(s->blk)) {

            error_report("Device needs media, but drive is empty");

            return -1;

        }

        if (blk_is_read_only(blk)) {

            error_report("Can't use a read-only drive");

            return -1;

        }

        blk_set_dev_ops(blk, &ide_hd_block_ops, s);

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

    blk_iostatus_enable(blk);

    return 0;

}
