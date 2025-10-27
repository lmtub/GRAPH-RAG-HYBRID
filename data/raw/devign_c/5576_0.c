void ide_init_drive(IDEState *s, DriveInfo *dinfo, const char *version)

{

    int cylinders, heads, secs;

    uint64_t nb_sectors;



    if (dinfo && dinfo->bdrv) {

        s->bs = dinfo->bdrv;

        bdrv_get_geometry(s->bs, &nb_sectors);

        bdrv_guess_geometry(s->bs, &cylinders, &heads, &secs);

        s->cylinders = cylinders;

        s->heads = heads;

        s->sectors = secs;

        s->nb_sectors = nb_sectors;

        /* The SMART values should be preserved across power cycles

           but they aren't.  */

        s->smart_enabled = 1;

        s->smart_autosave = 1;

        s->smart_errors = 0;

        s->smart_selftest_count = 0;

        if (bdrv_get_type_hint(s->bs) == BDRV_TYPE_CDROM) {

            s->is_cdrom = 1;

            bdrv_set_change_cb(s->bs, cdrom_change_cb, s);

        }

        strncpy(s->drive_serial_str, drive_get_serial(s->bs),

                sizeof(s->drive_serial_str));

    }

    if (strlen(s->drive_serial_str) == 0)

        snprintf(s->drive_serial_str, sizeof(s->drive_serial_str),

                 "QM%05d", s->drive_serial);

    if (version) {

        pstrcpy(s->version, sizeof(s->version), version);

    } else {

        pstrcpy(s->version, sizeof(s->version), QEMU_VERSION);

    }

    ide_reset(s);

}
