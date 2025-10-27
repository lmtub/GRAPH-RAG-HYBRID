static BlkverifyAIOCB *blkverify_aio_get(BlockDriverState *bs, bool is_write,

                                         int64_t sector_num, QEMUIOVector *qiov,

                                         int nb_sectors,

                                         BlockCompletionFunc *cb,

                                         void *opaque)

{

    BlkverifyAIOCB *acb = qemu_aio_get(&blkverify_aiocb_info, bs, cb, opaque);



    acb->is_write = is_write;

    acb->sector_num = sector_num;

    acb->nb_sectors = nb_sectors;

    acb->ret = -EINPROGRESS;

    acb->done = 0;

    acb->qiov = qiov;

    acb->buf = NULL;

    acb->verify = NULL;

    return acb;

}
