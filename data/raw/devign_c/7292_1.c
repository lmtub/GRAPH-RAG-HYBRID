int coroutine_fn bdrv_co_flush(BlockDriverState *bs)
{
    int ret;
    BdrvTrackedRequest req;
    if (!bs || !bdrv_is_inserted(bs) || bdrv_is_read_only(bs) ||
        bdrv_is_sg(bs)) {
        return 0;
    }
    tracked_request_begin(&req, bs, 0, 0, BDRV_TRACKED_FLUSH);
    int current_gen = bs->write_gen;
    /* Wait until any previous flushes are completed */
    while (bs->flush_started_gen != bs->flushed_gen) {
        qemu_co_queue_wait(&bs->flush_queue);
    }
    bs->flush_started_gen = current_gen;
    /* Write back all layers by calling one driver function */
    if (bs->drv->bdrv_co_flush) {
        ret = bs->drv->bdrv_co_flush(bs);
        goto out;
    }
    /* Write back cached data to the OS even with cache=unsafe */
    BLKDBG_EVENT(bs->file, BLKDBG_FLUSH_TO_OS);
    if (bs->drv->bdrv_co_flush_to_os) {
        ret = bs->drv->bdrv_co_flush_to_os(bs);
        if (ret < 0) {
            goto out;
        }
    }
    /* But don't actually force it to the disk with cache=unsafe */
    if (bs->open_flags & BDRV_O_NO_FLUSH) {
        goto flush_parent;
    }
    /* Check if we really need to flush anything */
    if (bs->flushed_gen == current_gen) {
        goto flush_parent;
    }
    BLKDBG_EVENT(bs->file, BLKDBG_FLUSH_TO_DISK);
    if (bs->drv->bdrv_co_flush_to_disk) {
        ret = bs->drv->bdrv_co_flush_to_disk(bs);
    } else if (bs->drv->bdrv_aio_flush) {
        BlockAIOCB *acb;
        CoroutineIOCompletion co = {
            .coroutine = qemu_coroutine_self(),
        };
        acb = bs->drv->bdrv_aio_flush(bs, bdrv_co_io_em_complete, &co);
        if (acb == NULL) {
            ret = -EIO;
        } else {
            qemu_coroutine_yield();
            ret = co.ret;
        }
    } else {
        /*
         * Some block drivers always operate in either writethrough or unsafe
         * mode and don't support bdrv_flush therefore. Usually qemu doesn't
         * know how the server works (because the behaviour is hardcoded or
         * depends on server-side configuration), so we can't ensure that
         * everything is safe on disk. Returning an error doesn't work because
         * that would break guests even if the server operates in writethrough
         * mode.
         *
         * Let's hope the user knows what he's doing.
         */
        ret = 0;
    }
    if (ret < 0) {
        goto out;
    }
    /* Now flush the underlying protocol.  It will also have BDRV_O_NO_FLUSH
     * in the case of cache=unsafe, so there are no useless flushes.
     */
flush_parent:
    ret = bs->file ? bdrv_co_flush(bs->file->bs) : 0;
out:
    /* Notify any pending flushes that we have completed */
    bs->flushed_gen = current_gen;
    qemu_co_queue_restart_all(&bs->flush_queue);
    tracked_request_end(&req);
    return ret;
}