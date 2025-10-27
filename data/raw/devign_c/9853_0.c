int coroutine_fn bdrv_co_readv(BlockDriverState *bs, int64_t sector_num,

    int nb_sectors, QEMUIOVector *qiov)

{

    trace_bdrv_co_readv(bs, sector_num, nb_sectors);



    return bdrv_co_do_readv(bs, sector_num, nb_sectors, qiov, 0);

}
