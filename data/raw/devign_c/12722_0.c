static QCowAIOCB *qcow_aio_setup(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque, int is_write)

{

    QCowAIOCB *acb;



    acb = qemu_aio_get(&qcow_aio_pool, bs, cb, opaque);

    if (!acb)

        return NULL;

    acb->hd_aiocb = NULL;

    acb->sector_num = sector_num;

    acb->qiov = qiov;

    if (qiov->niov > 1) {

        acb->buf = acb->orig_buf = qemu_blockalign(bs, qiov->size);

        if (is_write)

            qemu_iovec_to_buffer(qiov, acb->buf);

    } else {

        acb->buf = (uint8_t *)qiov->iov->iov_base;

    }

    acb->nb_sectors = nb_sectors;

    acb->n = 0;

    acb->cluster_offset = 0;

    acb->l2meta.nb_clusters = 0;

    LIST_INIT(&acb->l2meta.dependent_requests);

    return acb;

}
