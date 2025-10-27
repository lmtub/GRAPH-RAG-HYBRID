static int64_t coroutine_fn bdrv_co_get_block_status(BlockDriverState *bs,

                                                     int64_t sector_num,

                                                     int nb_sectors, int *pnum,

                                                     BlockDriverState **file)

{

    int64_t total_sectors;

    int64_t n;

    int64_t ret, ret2;



    *file = NULL;

    total_sectors = bdrv_nb_sectors(bs);

    if (total_sectors < 0) {

        return total_sectors;

    }



    if (sector_num >= total_sectors) {

        *pnum = 0;

        return BDRV_BLOCK_EOF;

    }

    if (!nb_sectors) {

        *pnum = 0;

        return 0;

    }



    n = total_sectors - sector_num;

    if (n < nb_sectors) {

        nb_sectors = n;

    }



    if (!bs->drv->bdrv_co_get_block_status) {

        *pnum = nb_sectors;

        ret = BDRV_BLOCK_DATA | BDRV_BLOCK_ALLOCATED;

        if (sector_num + nb_sectors == total_sectors) {

            ret |= BDRV_BLOCK_EOF;

        }

        if (bs->drv->protocol_name) {

            ret |= BDRV_BLOCK_OFFSET_VALID | (sector_num * BDRV_SECTOR_SIZE);

            *file = bs;

        }

        return ret;

    }



    bdrv_inc_in_flight(bs);

    ret = bs->drv->bdrv_co_get_block_status(bs, sector_num, nb_sectors, pnum,

                                            file);

    if (ret < 0) {

        *pnum = 0;

        goto out;

    }



    if (ret & BDRV_BLOCK_RAW) {

        assert(ret & BDRV_BLOCK_OFFSET_VALID && *file);

        ret = bdrv_co_get_block_status(*file, ret >> BDRV_SECTOR_BITS,

                                       *pnum, pnum, file);

        goto out;

    }



    if (ret & (BDRV_BLOCK_DATA | BDRV_BLOCK_ZERO)) {

        ret |= BDRV_BLOCK_ALLOCATED;

    } else {

        if (bdrv_unallocated_blocks_are_zero(bs)) {

            ret |= BDRV_BLOCK_ZERO;

        } else if (bs->backing) {

            BlockDriverState *bs2 = bs->backing->bs;

            int64_t nb_sectors2 = bdrv_nb_sectors(bs2);

            if (nb_sectors2 >= 0 && sector_num >= nb_sectors2) {

                ret |= BDRV_BLOCK_ZERO;

            }

        }

    }



    if (*file && *file != bs &&

        (ret & BDRV_BLOCK_DATA) && !(ret & BDRV_BLOCK_ZERO) &&

        (ret & BDRV_BLOCK_OFFSET_VALID)) {

        BlockDriverState *file2;

        int file_pnum;



        ret2 = bdrv_co_get_block_status(*file, ret >> BDRV_SECTOR_BITS,

                                        *pnum, &file_pnum, &file2);

        if (ret2 >= 0) {

            /* Ignore errors.  This is just providing extra information, it

             * is useful but not necessary.

             */

            if (ret2 & BDRV_BLOCK_EOF &&

                (!file_pnum || ret2 & BDRV_BLOCK_ZERO)) {

                /*

                 * It is valid for the format block driver to read

                 * beyond the end of the underlying file's current

                 * size; such areas read as zero.

                 */

                ret |= BDRV_BLOCK_ZERO;

            } else {

                /* Limit request to the range reported by the protocol driver */

                *pnum = file_pnum;

                ret |= (ret2 & BDRV_BLOCK_ZERO);

            }

        }

    }



out:

    bdrv_dec_in_flight(bs);

    if (ret >= 0 && sector_num + *pnum == total_sectors) {

        ret |= BDRV_BLOCK_EOF;

    }

    return ret;

}
