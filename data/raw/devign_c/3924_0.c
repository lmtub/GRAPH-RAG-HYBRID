BlockAIOCB *bdrv_aio_write_zeroes(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, BdrvRequestFlags flags,

        BlockCompletionFunc *cb, void *opaque)

{

    trace_bdrv_aio_write_zeroes(bs, sector_num, nb_sectors, flags, opaque);



    return bdrv_co_aio_rw_vector(bs, sector_num, NULL, nb_sectors,

                                 BDRV_REQ_ZERO_WRITE | flags,

                                 cb, opaque, true);

}
