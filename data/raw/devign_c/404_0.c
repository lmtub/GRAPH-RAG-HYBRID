BlockDriverAIOCB *paio_submit(BlockDriverState *bs, int fd,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque, int type)

{

    struct qemu_paiocb *acb;



    acb = qemu_aio_get(&raw_aio_pool, bs, cb, opaque);

    if (!acb)

        return NULL;

    acb->aio_type = type;

    acb->aio_fildes = fd;

    acb->ev_signo = SIGUSR2;

    acb->async_context_id = get_async_context_id();



    if (qiov) {

        acb->aio_iov = qiov->iov;

        acb->aio_niov = qiov->niov;

    }

    acb->aio_nbytes = nb_sectors * 512;

    acb->aio_offset = sector_num * 512;



    acb->next = posix_aio_state->first_aio;

    posix_aio_state->first_aio = acb;



    trace_paio_submit(acb, opaque, sector_num, nb_sectors, type);

    qemu_paio_submit(acb);

    return &acb->common;

}
