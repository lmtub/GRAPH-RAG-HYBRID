int coroutine_fn bdrv_co_pdiscard(BlockDriverState *bs, int64_t offset,

                                  int count)

{

    BdrvTrackedRequest req;

    int max_pdiscard, ret;

    int head, align;



    if (!bs->drv) {

        return -ENOMEDIUM;

    }



    ret = bdrv_check_byte_request(bs, offset, count);

    if (ret < 0) {

        return ret;

    } else if (bs->read_only) {

        return -EPERM;

    }

    assert(!(bs->open_flags & BDRV_O_INACTIVE));



    /* Do nothing if disabled.  */

    if (!(bs->open_flags & BDRV_O_UNMAP)) {

        return 0;

    }



    if (!bs->drv->bdrv_co_pdiscard && !bs->drv->bdrv_aio_pdiscard) {

        return 0;

    }



    /* Discard is advisory, so ignore any unaligned head or tail */

    align = MAX(bs->bl.pdiscard_alignment, bs->bl.request_alignment);

    assert(align % bs->bl.request_alignment == 0);

    head = offset % align;

    if (head) {

        head = MIN(count, align - head);

        count -= head;

        offset += head;

    }

    count = QEMU_ALIGN_DOWN(count, align);

    if (!count) {

        return 0;

    }



    bdrv_inc_in_flight(bs);

    tracked_request_begin(&req, bs, offset, count, BDRV_TRACKED_DISCARD);



    ret = notifier_with_return_list_notify(&bs->before_write_notifiers, &req);

    if (ret < 0) {

        goto out;

    }



    max_pdiscard = QEMU_ALIGN_DOWN(MIN_NON_ZERO(bs->bl.max_pdiscard, INT_MAX),

                                   align);

    assert(max_pdiscard);



    while (count > 0) {

        int ret;

        int num = MIN(count, max_pdiscard);



        if (bs->drv->bdrv_co_pdiscard) {

            ret = bs->drv->bdrv_co_pdiscard(bs, offset, num);

        } else {

            BlockAIOCB *acb;

            CoroutineIOCompletion co = {

                .coroutine = qemu_coroutine_self(),

            };



            acb = bs->drv->bdrv_aio_pdiscard(bs, offset, num,

                                             bdrv_co_io_em_complete, &co);

            if (acb == NULL) {

                ret = -EIO;

                goto out;

            } else {

                qemu_coroutine_yield();

                ret = co.ret;

            }

        }

        if (ret && ret != -ENOTSUP) {

            goto out;

        }



        offset += num;

        count -= num;

    }

    ret = 0;

out:

    ++bs->write_gen;

    bdrv_set_dirty(bs, req.offset >> BDRV_SECTOR_BITS,

                   req.bytes >> BDRV_SECTOR_BITS);

    tracked_request_end(&req);

    bdrv_dec_in_flight(bs);

    return ret;

}
