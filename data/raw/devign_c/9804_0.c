static void ide_init2(IDEState *ide_state,

                      BlockDriverState *hd0, BlockDriverState *hd1,

                      qemu_irq irq)

{

    IDEState *s;

    static int drive_serial = 1;

    int i, cylinders, heads, secs;

    uint64_t nb_sectors;



    for(i = 0; i < 2; i++) {

        s = ide_state + i;

        s->io_buffer = qemu_memalign(512, IDE_DMA_BUF_SECTORS*512 + 4);

        if (i == 0)

            s->bs = hd0;

        else

            s->bs = hd1;

        if (s->bs) {

            bdrv_get_geometry(s->bs, &nb_sectors);

            bdrv_guess_geometry(s->bs, &cylinders, &heads, &secs);

            s->cylinders = cylinders;

            s->heads = heads;

            s->sectors = secs;

            s->nb_sectors = nb_sectors;



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

        s->irq = irq;

        s->sector_write_timer = qemu_new_timer(vm_clock,

                                               ide_sector_write_timer_cb, s);

        ide_reset(s);

    }

}
