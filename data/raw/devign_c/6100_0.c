static int coroutine_fn bdrv_co_writev_em(BlockDriverState *bs,

                                         int64_t sector_num, int nb_sectors,

                                         QEMUIOVector *iov)

{

    return bdrv_co_io_em(bs, sector_num, nb_sectors, iov, true);

}
