static BlockDriverAIOCB *raw_aio_writev(BlockDriverState *bs,

    int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

    BlockDriverCompletionFunc *cb, void *opaque)

{

    return bdrv_aio_writev(bs->file, sector_num, qiov, nb_sectors, cb, opaque);

}
