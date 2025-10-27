static int coroutine_fn bdrv_aligned_pwritev(BlockDriverState *bs,

    BdrvTrackedRequest *req, int64_t offset, unsigned int bytes,

    int64_t align, QEMUIOVector *qiov, int flags)

{

    BlockDriver *drv = bs->drv;

    bool waited;

    int ret;



    int64_t start_sector = offset >> BDRV_SECTOR_BITS;

    int64_t end_sector = DIV_ROUND_UP(offset + bytes, BDRV_SECTOR_SIZE);



    assert(is_power_of_2(align));

    assert((offset & (align - 1)) == 0);

    assert((bytes & (align - 1)) == 0);

    assert(!qiov || bytes == qiov->size);

    assert((bs->open_flags & BDRV_O_NO_IO) == 0);

    assert(!(flags & ~BDRV_REQ_MASK));



    waited = wait_serialising_requests(req);

    assert(!waited || !req->serialising);

    assert(req->overlap_offset <= offset);

    assert(offset + bytes <= req->overlap_offset + req->overlap_bytes);



    ret = notifier_with_return_list_notify(&bs->before_write_notifiers, req);



    if (!ret && bs->detect_zeroes != BLOCKDEV_DETECT_ZEROES_OPTIONS_OFF &&

        !(flags & BDRV_REQ_ZERO_WRITE) && drv->bdrv_co_pwrite_zeroes &&

        qemu_iovec_is_zero(qiov)) {

        flags |= BDRV_REQ_ZERO_WRITE;

        if (bs->detect_zeroes == BLOCKDEV_DETECT_ZEROES_OPTIONS_UNMAP) {

            flags |= BDRV_REQ_MAY_UNMAP;

        }

    }



    if (ret < 0) {

        /* Do nothing, write notifier decided to fail this request */

    } else if (flags & BDRV_REQ_ZERO_WRITE) {

        bdrv_debug_event(bs, BLKDBG_PWRITEV_ZERO);

        ret = bdrv_co_do_pwrite_zeroes(bs, offset, bytes, flags);

    } else {

        bdrv_debug_event(bs, BLKDBG_PWRITEV);

        ret = bdrv_driver_pwritev(bs, offset, bytes, qiov, flags);

    }

    bdrv_debug_event(bs, BLKDBG_PWRITEV_DONE);




    bdrv_set_dirty(bs, start_sector, end_sector - start_sector);



    if (bs->wr_highest_offset < offset + bytes) {

        bs->wr_highest_offset = offset + bytes;

    }



    if (ret >= 0) {

        bs->total_sectors = MAX(bs->total_sectors, end_sector);

    }



    return ret;

}