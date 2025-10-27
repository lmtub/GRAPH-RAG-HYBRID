BlockDriverAIOCB *laio_submit(BlockDriverState *bs, void *aio_ctx, int fd,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque, int type)

{

    struct qemu_laio_state *s = aio_ctx;

    struct qemu_laiocb *laiocb;

    struct iocb *iocbs;

    off_t offset = sector_num * 512;



    laiocb = qemu_aio_get(&laio_pool, bs, cb, opaque);

    if (!laiocb)

        return NULL;

    laiocb->nbytes = nb_sectors * 512;

    laiocb->ctx = s;

    laiocb->ret = -EINPROGRESS;

    laiocb->async_context_id = get_async_context_id();



    iocbs = &laiocb->iocb;



    switch (type) {

    case QEMU_AIO_WRITE:

        io_prep_pwritev(iocbs, fd, qiov->iov, qiov->niov, offset);

	break;

    case QEMU_AIO_READ:

        io_prep_preadv(iocbs, fd, qiov->iov, qiov->niov, offset);

	break;

    default:

        fprintf(stderr, "%s: invalid AIO request type 0x%x.\n",

                        __func__, type);

        goto out_free_aiocb;

    }

    io_set_eventfd(&laiocb->iocb, s->efd);

    s->count++;



    if (io_submit(s->ctx, 1, &iocbs) < 0)

        goto out_dec_count;

    return &laiocb->common;



out_free_aiocb:

    qemu_aio_release(laiocb);

out_dec_count:

    s->count--;

    return NULL;

}
