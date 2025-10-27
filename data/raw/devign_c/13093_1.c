static BlockAIOCB *hdev_aio_ioctl(BlockDriverState *bs,
        unsigned long int req, void *buf,
        BlockCompletionFunc *cb, void *opaque)
{
    BDRVRawState *s = bs->opaque;
    RawPosixAIOData *acb;
    ThreadPool *pool;
    if (fd_open(bs) < 0)
        return NULL;
    acb = g_new(RawPosixAIOData, 1);
    acb->bs = bs;
    acb->aio_type = QEMU_AIO_IOCTL;
    acb->aio_fildes = s->fd;
    acb->aio_offset = 0;
    acb->aio_ioctl_buf = buf;
    acb->aio_ioctl_cmd = req;
    pool = aio_get_thread_pool(bdrv_get_aio_context(bs));
    return thread_pool_submit_aio(pool, aio_worker, acb, cb, opaque);