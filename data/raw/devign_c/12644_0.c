static int64_t coroutine_fn bdrv_co_get_block_status(BlockDriverState *bs,

                                                     int64_t sector_num,

                                                     int nb_sectors, int *pnum)

{

    int64_t length;

    int64_t n;

    int64_t ret, ret2;



    length = bdrv_getlength(bs);

    if (length < 0) {

        return length;

    }



    if (sector_num >= (length >> BDRV_SECTOR_BITS)) {

        *pnum = 0;

        return 0;

    }



    n = bs->total_sectors - sector_num;

    if (n < nb_sectors) {

        nb_sectors = n;

    }



    if (!bs->drv->bdrv_co_get_block_status) {

        *pnum = nb_sectors;

        ret = BDRV_BLOCK_DATA;

        if (bs->drv->protocol_name) {

            ret |= BDRV_BLOCK_OFFSET_VALID | (sector_num * BDRV_SECTOR_SIZE);

        }

        return ret;

    }



    ret = bs->drv->bdrv_co_get_block_status(bs, sector_num, nb_sectors, pnum);

    if (ret < 0) {

        *pnum = 0;

        return ret;

    }



    if (ret & BDRV_BLOCK_RAW) {

        assert(ret & BDRV_BLOCK_OFFSET_VALID);

        return bdrv_get_block_status(bs->file, ret >> BDRV_SECTOR_BITS,

                                     *pnum, pnum);

    }



    if (!(ret & BDRV_BLOCK_DATA)) {

        if (bdrv_has_zero_init(bs)) {

            ret |= BDRV_BLOCK_ZERO;

        } else if (bs->backing_hd) {

            BlockDriverState *bs2 = bs->backing_hd;

            int64_t length2 = bdrv_getlength(bs2);

            if (length2 >= 0 && sector_num >= (length2 >> BDRV_SECTOR_BITS)) {

                ret |= BDRV_BLOCK_ZERO;

            }

        }

    }



    if (bs->file &&

        (ret & BDRV_BLOCK_DATA) && !(ret & BDRV_BLOCK_ZERO) &&

        (ret & BDRV_BLOCK_OFFSET_VALID)) {

        ret2 = bdrv_co_get_block_status(bs->file, ret >> BDRV_SECTOR_BITS,

                                        *pnum, pnum);

        if (ret2 >= 0) {

            /* Ignore errors.  This is just providing extra information, it

             * is useful but not necessary.

             */

            ret |= (ret2 & BDRV_BLOCK_ZERO);

        }

    }



    return ret;

}
