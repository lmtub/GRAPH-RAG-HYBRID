static int coroutine_fn bdrv_driver_pwritev(BlockDriverState *bs,
                                            uint64_t offset, uint64_t bytes,
                                            QEMUIOVector *qiov, int flags)
{
    BlockDriver *drv = bs->drv;
    int64_t sector_num;
    unsigned int nb_sectors;
    int ret;
    if (drv->bdrv_co_pwritev) {
        ret = drv->bdrv_co_pwritev(bs, offset, bytes, qiov,
                                   flags & bs->supported_write_flags);
        flags &= ~bs->supported_write_flags;
        goto emulate_flags;
    }
    sector_num = offset >> BDRV_SECTOR_BITS;
    nb_sectors = bytes >> BDRV_SECTOR_BITS;
    assert((offset & (BDRV_SECTOR_SIZE - 1)) == 0);
    assert((bytes & (BDRV_SECTOR_SIZE - 1)) == 0);
    assert((bytes >> BDRV_SECTOR_BITS) <= BDRV_REQUEST_MAX_SECTORS);
    if (drv->bdrv_co_writev_flags) {
        ret = drv->bdrv_co_writev_flags(bs, sector_num, nb_sectors, qiov,
                                        flags & bs->supported_write_flags);
        flags &= ~bs->supported_write_flags;
    } else if (drv->bdrv_co_writev) {
        assert(!bs->supported_write_flags);
        ret = drv->bdrv_co_writev(bs, sector_num, nb_sectors, qiov);
    } else {
        BlockAIOCB *acb;
        CoroutineIOCompletion co = {
            .coroutine = qemu_coroutine_self(),
        };
        acb = bs->drv->bdrv_aio_writev(bs, sector_num, qiov, nb_sectors,
                                       bdrv_co_io_em_complete, &co);
        if (acb == NULL) {
            ret = -EIO;
        } else {
            qemu_coroutine_yield();
            ret = co.ret;
        }
    }
emulate_flags:
    if (ret == 0 && (flags & BDRV_REQ_FUA)) {
        ret = bdrv_co_flush(bs);
    }
    return ret;
}