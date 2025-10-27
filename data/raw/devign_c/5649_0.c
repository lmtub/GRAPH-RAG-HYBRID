int coroutine_fn bdrv_co_copy_on_readv(BlockDriverState *bs,

    int64_t sector_num, int nb_sectors, QEMUIOVector *qiov)

{

    trace_bdrv_co_copy_on_readv(bs, sector_num, nb_sectors);



    return bdrv_co_do_readv(bs, sector_num, nb_sectors, qiov,

                            BDRV_REQ_COPY_ON_READ);

}
