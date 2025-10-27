static BlockDriverAIOCB *paio_ioctl(BlockDriverState *bs, int fd,

        unsigned long int req, void *buf,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    RawPosixAIOData *acb = g_slice_new(RawPosixAIOData);



    acb->bs = bs;

    acb->aio_type = QEMU_AIO_IOCTL;

    acb->aio_fildes = fd;

    acb->aio_offset = 0;

    acb->aio_ioctl_buf = buf;

    acb->aio_ioctl_cmd = req;



    return thread_pool_submit_aio(aio_worker, acb, cb, opaque);

}
