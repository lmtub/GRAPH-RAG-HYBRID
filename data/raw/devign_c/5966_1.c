static int64_t coroutine_fn raw_co_get_block_status(BlockDriverState *bs,

                                                    int64_t sector_num,

                                                    int nb_sectors, int *pnum)

{

    off_t start, data = 0, hole = 0;

    int64_t total_size;

    int ret;



    ret = fd_open(bs);

    if (ret < 0) {

        return ret;

    }



    start = sector_num * BDRV_SECTOR_SIZE;

    total_size = bdrv_getlength(bs);

    if (total_size < 0) {

        return total_size;

    } else if (start >= total_size) {

        *pnum = 0;

        return 0;

    } else if (start + nb_sectors * BDRV_SECTOR_SIZE > total_size) {

        nb_sectors = DIV_ROUND_UP(total_size - start, BDRV_SECTOR_SIZE);

    }



    ret = find_allocation(bs, start, &data, &hole);

    if (ret == -ENXIO) {

        /* Trailing hole */

        *pnum = nb_sectors;

        ret = BDRV_BLOCK_ZERO;

    } else if (ret < 0) {

        /* No info available, so pretend there are no holes */

        *pnum = nb_sectors;

        ret = BDRV_BLOCK_DATA;

    } else if (data == start) {

        /* On a data extent, compute sectors to the end of the extent.  */

        *pnum = MIN(nb_sectors, (hole - start) / BDRV_SECTOR_SIZE);

        ret = BDRV_BLOCK_DATA;

    } else {

        /* On a hole, compute sectors to the beginning of the next extent.  */

        assert(hole == start);

        *pnum = MIN(nb_sectors, (data - start) / BDRV_SECTOR_SIZE);

        ret = BDRV_BLOCK_ZERO;

    }

    return ret | BDRV_BLOCK_OFFSET_VALID | start;

}
