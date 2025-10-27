static int coroutine_fn bdrv_co_do_readv(BlockDriverState *bs,

    int64_t sector_num, int nb_sectors, QEMUIOVector *qiov,

    BdrvRequestFlags flags)

{

    if (nb_sectors < 0 || nb_sectors > (UINT_MAX >> BDRV_SECTOR_BITS)) {

        return -EINVAL;

    }



    return bdrv_co_do_preadv(bs, sector_num << BDRV_SECTOR_BITS,

                             nb_sectors << BDRV_SECTOR_BITS, qiov, flags);

}
