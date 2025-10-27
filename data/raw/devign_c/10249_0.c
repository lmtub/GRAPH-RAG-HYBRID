static BlockDriverAIOCB *curl_aio_readv(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    CURLAIOCB *acb;



    acb = qemu_aio_get(&curl_aiocb_info, bs, cb, opaque);



    acb->qiov = qiov;

    acb->sector_num = sector_num;

    acb->nb_sectors = nb_sectors;



    acb->bh = qemu_bh_new(curl_readv_bh_cb, acb);

    qemu_bh_schedule(acb->bh);

    return &acb->common;

}
