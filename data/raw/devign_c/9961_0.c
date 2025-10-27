static BlockDriverAIOCB *raw_aio_writev(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    RawAIOCB *acb;



    acb = raw_aio_setup(bs, sector_num, qiov, nb_sectors, cb, opaque);

    if (!acb)

        return NULL;

    if (qemu_paio_write(&acb->aiocb) < 0) {

        raw_aio_remove(acb);

        return NULL;

    }

    return &acb->common;

}
