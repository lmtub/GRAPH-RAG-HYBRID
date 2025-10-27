BlockAIOCB *ide_issue_trim(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockCompletionFunc *cb, void *opaque)

{

    TrimAIOCB *iocb;



    iocb = qemu_aio_get(&trim_aiocb_info, bs, cb, opaque);

    iocb->bh = qemu_bh_new(ide_trim_bh_cb, iocb);

    iocb->ret = 0;

    iocb->qiov = qiov;

    iocb->i = -1;

    iocb->j = 0;

    ide_issue_trim_cb(iocb, 0);

    return &iocb->common;

}
