void ide_init2(IDEBus *bus, DriveInfo *hd0, DriveInfo *hd1,

               qemu_irq irq)

{

    IDEState *s;

    static int drive_serial = 1;

    int i, cylinders, heads, secs;

    uint64_t nb_sectors;



    for(i = 0; i < 2; i++) {

        s = bus->ifs + i;

        s->bus = bus;

        s->unit = i;

        if (i == 0 && hd0)

            s->bs = hd0->bdrv;

        if (i == 1 && hd1)

            s->bs = hd1->bdrv;

        s->io_buffer = qemu_blockalign(s->bs, IDE_DMA_BUF_SECTORS*512 + 4);

        if (s->bs) {

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

	    s->smart_selftest_data = qemu_blockalign(s->bs, 512);

            if (bdrv_get_type_hint(s->bs) == BDRV_TYPE_CDROM) {

                s->is_cdrom = 1;

		bdrv_set_change_cb(s->bs, cdrom_change_cb, s);

            }

        }

        s->drive_serial = drive_serial++;

        strncpy(s->drive_serial_str, drive_get_serial(s->bs),

                sizeof(s->drive_serial_str));

        if (strlen(s->drive_serial_str) == 0)

            snprintf(s->drive_serial_str, sizeof(s->drive_serial_str),

                    "QM%05d", s->drive_serial);

        s->sector_write_timer = qemu_new_timer(vm_clock,

                                               ide_sector_write_timer_cb, s);

        ide_reset(s);

    }

    bus->irq = irq;

}
