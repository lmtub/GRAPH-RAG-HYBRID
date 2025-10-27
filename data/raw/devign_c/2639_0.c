static BlockAIOCB *raw_aio_writev(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockCompletionFunc *cb, void *opaque)

{

    return raw_aio_submit(bs, sector_num, qiov, nb_sectors,

                          cb, opaque, QEMU_AIO_WRITE);

}
