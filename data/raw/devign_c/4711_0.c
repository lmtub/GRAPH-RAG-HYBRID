static int coroutine_fn bdrv_co_do_writev(BlockDriverState *bs,

    int64_t sector_num, int nb_sectors, QEMUIOVector *qiov,

    BdrvRequestFlags flags)

{

    if (nb_sectors < 0 || nb_sectors > BDRV_REQUEST_MAX_SECTORS) {

        return -EINVAL;

    }



    return bdrv_co_do_pwritev(bs, sector_num << BDRV_SECTOR_BITS,

                              nb_sectors << BDRV_SECTOR_BITS, qiov, flags);

}
