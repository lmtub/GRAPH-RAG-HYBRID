int ide_init_drive(IDEState *s, BlockDriverState *bs,

                   const char *version, const char *serial)

{

    int cylinders, heads, secs;

    uint64_t nb_sectors;



    s->bs = bs;

    bdrv_get_geometry(bs, &nb_sectors);

    bdrv_guess_geometry(bs, &cylinders, &heads, &secs);

    if (cylinders < 1 || cylinders > 16383) {

        error_report("cyls must be between 1 and 16383");



    if (heads < 1 || heads > 16) {

        error_report("heads must be between 1 and 16");



    if (secs < 1 || secs > 63) {

        error_report("secs must be between 1 and 63");



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

    if (bdrv_get_type_hint(bs) == BDRV_TYPE_CDROM) {

        s->drive_kind = IDE_CD;

        bdrv_set_change_cb(bs, cdrom_change_cb, s);

    } else {





        if (bdrv_is_read_only(bs)) {

            error_report("Can't use a read-only drive");




    if (serial) {

        strncpy(s->drive_serial_str, serial, sizeof(s->drive_serial_str));

    } else {

        snprintf(s->drive_serial_str, sizeof(s->drive_serial_str),

                 "QM%05d", s->drive_serial);


    if (version) {

        pstrcpy(s->version, sizeof(s->version), version);

    } else {

        pstrcpy(s->version, sizeof(s->version), QEMU_VERSION);


    ide_reset(s);

    bdrv_set_removable(bs, s->drive_kind == IDE_CD);

    return 0;
