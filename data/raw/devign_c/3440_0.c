static BlockDriverAIOCB *raw_aio_readv(BlockDriverState *bs,

                         int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

                         BlockDriverCompletionFunc *cb, void *opaque)

{

    BDRVRawState *s = bs->opaque;

    return paio_submit(bs, s->hfile, sector_num, qiov, nb_sectors,

                       cb, opaque, QEMU_AIO_READ);

}
