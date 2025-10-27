static int m25p80_init(SSISlave *ss)

{

    DriveInfo *dinfo;

    Flash *s = M25P80(ss);

    M25P80Class *mc = M25P80_GET_CLASS(s);



    s->pi = mc->pi;



    s->size = s->pi->sector_size * s->pi->n_sectors;

    s->dirty_page = -1;

    s->storage = qemu_blockalign(s->bdrv, s->size);



    dinfo = drive_get_next(IF_MTD);



    if (dinfo) {

        DB_PRINT_L(0, "Binding to IF_MTD drive\n");

        s->bdrv = blk_bs(blk_by_legacy_dinfo(dinfo));



        /* FIXME: Move to late init */

        if (bdrv_read(s->bdrv, 0, s->storage, DIV_ROUND_UP(s->size,

                                                    BDRV_SECTOR_SIZE))) {

            fprintf(stderr, "Failed to initialize SPI flash!\n");

            return 1;

        }

    } else {

        DB_PRINT_L(0, "No BDRV - binding to RAM\n");

        memset(s->storage, 0xFF, s->size);

    }



    return 0;

}
