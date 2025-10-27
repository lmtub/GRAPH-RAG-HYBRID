BlockDriverAIOCB *paio_ioctl(BlockDriverState *bs, int fd,

        unsigned long int req, void *buf,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    struct qemu_paiocb *acb;



    acb = qemu_aio_get(&raw_aio_pool, bs, cb, opaque);

    if (!acb)

        return NULL;

    acb->aio_type = QEMU_AIO_IOCTL;

    acb->aio_fildes = fd;

    acb->ev_signo = SIGUSR2;

    acb->async_context_id = get_async_context_id();

    acb->aio_offset = 0;

    acb->aio_ioctl_buf = buf;

    acb->aio_ioctl_cmd = req;



    acb->next = posix_aio_state->first_aio;

    posix_aio_state->first_aio = acb;



    qemu_paio_submit(acb);

    return &acb->common;

}
