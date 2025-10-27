static int guess_disk_lchs(BlockDriverState *bs,

                           int *pcylinders, int *pheads, int *psectors)

{

    uint8_t buf[BDRV_SECTOR_SIZE];

    int i, heads, sectors, cylinders;

    struct partition *p;

    uint32_t nr_sects;

    uint64_t nb_sectors;



    bdrv_get_geometry(bs, &nb_sectors);



    /**

     * The function will be invoked during startup not only in sync I/O mode,

     * but also in async I/O mode. So the I/O throttling function has to

     * be disabled temporarily here, not permanently.

     */

    if (bdrv_read_unthrottled(bs, 0, buf, 1) < 0) {

        return -1;

    }

    /* test msdos magic */

    if (buf[510] != 0x55 || buf[511] != 0xaa) {

        return -1;

    }

    for (i = 0; i < 4; i++) {

        p = ((struct partition *)(buf + 0x1be)) + i;

        nr_sects = le32_to_cpu(p->nr_sects);

        if (nr_sects && p->end_head) {

            /* We make the assumption that the partition terminates on

               a cylinder boundary */

            heads = p->end_head + 1;

            sectors = p->end_sector & 63;

            if (sectors == 0) {

                continue;

            }

            cylinders = nb_sectors / (heads * sectors);

            if (cylinders < 1 || cylinders > 16383) {

                continue;

            }

            *pheads = heads;

            *psectors = sectors;

            *pcylinders = cylinders;

            trace_hd_geometry_lchs_guess(bs, cylinders, heads, sectors);

            return 0;

        }

    }

    return -1;

}
