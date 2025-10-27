static int m25p80_init(SSISlave *ss)

{

    DriveInfo *dinfo;

    Flash *s = M25P80(ss);

    M25P80Class *mc = M25P80_GET_CLASS(s);



    s->pi = mc->pi;



    s->size = s->pi->sector_size * s->pi->n_sectors;

    s->dirty_page = -1;

    s->storage = blk_blockalign(s->blk, s->size);



    /* FIXME use a qdev drive property instead of drive_get_next() */

    dinfo = drive_get_next(IF_MTD);



    if (dinfo) {

        DB_PRINT_L(0, "Binding to IF_MTD drive\n");

        s->blk = blk_by_legacy_dinfo(dinfo);

        blk_attach_dev_nofail(s->blk, s);



        /* FIXME: Move to late init */

        if (blk_read(s->blk, 0, s->storage,

                     DIV_ROUND_UP(s->size, BDRV_SECTOR_SIZE))) {

            fprintf(stderr, "Failed to initialize SPI flash!\n");

            return 1;

        }

    } else {

        DB_PRINT_L(0, "No BDRV - binding to RAM\n");

        memset(s->storage, 0xFF, s->size);

    }



    return 0;

}
