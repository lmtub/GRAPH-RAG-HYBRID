static int coroutine_fn bdrv_aligned_pwritev(BlockDriverState *bs,

    int64_t offset, unsigned int bytes, QEMUIOVector *qiov, int flags)

{

    BlockDriver *drv = bs->drv;

    BdrvTrackedRequest req;

    int ret;



    int64_t sector_num = offset >> BDRV_SECTOR_BITS;

    unsigned int nb_sectors = bytes >> BDRV_SECTOR_BITS;



    assert((offset & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert((bytes & (BDRV_SECTOR_SIZE - 1)) == 0);



    if (bs->copy_on_read_in_flight) {

        wait_for_overlapping_requests(bs, offset, bytes);

    }



    tracked_request_begin(&req, bs, offset, bytes, true);



    ret = notifier_with_return_list_notify(&bs->before_write_notifiers, &req);



    if (ret < 0) {

        /* Do nothing, write notifier decided to fail this request */

    } else if (flags & BDRV_REQ_ZERO_WRITE) {

        ret = bdrv_co_do_write_zeroes(bs, sector_num, nb_sectors, flags);

    } else {

        ret = drv->bdrv_co_writev(bs, sector_num, nb_sectors, qiov);

    }



    if (ret == 0 && !bs->enable_write_cache) {

        ret = bdrv_co_flush(bs);

    }



    bdrv_set_dirty(bs, sector_num, nb_sectors);



    if (bs->wr_highest_sector < sector_num + nb_sectors - 1) {

        bs->wr_highest_sector = sector_num + nb_sectors - 1;

    }

    if (bs->growable && ret >= 0) {

        bs->total_sectors = MAX(bs->total_sectors, sector_num + nb_sectors);

    }



    tracked_request_end(&req);



    return ret;

}
