int coroutine_fn bdrv_co_discard(BlockDriverState *bs, int64_t sector_num,

                                 int nb_sectors)

{

    BdrvTrackedRequest req;

    int max_discard, ret;



    if (!bs->drv) {

        return -ENOMEDIUM;

    }



    ret = bdrv_check_request(bs, sector_num, nb_sectors);

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



    if (!bs->drv->bdrv_co_discard && !bs->drv->bdrv_aio_discard) {

        return 0;

    }



    tracked_request_begin(&req, bs, sector_num << BDRV_SECTOR_BITS,

                          nb_sectors << BDRV_SECTOR_BITS, BDRV_TRACKED_DISCARD);



    ret = notifier_with_return_list_notify(&bs->before_write_notifiers, &req);

    if (ret < 0) {

        goto out;

    }



    max_discard = MIN_NON_ZERO(bs->bl.max_pdiscard >> BDRV_SECTOR_BITS,

                               BDRV_REQUEST_MAX_SECTORS);

    while (nb_sectors > 0) {

        int ret;

        int num = nb_sectors;

        int discard_alignment = bs->bl.pdiscard_alignment >> BDRV_SECTOR_BITS;



        /* align request */

        if (discard_alignment &&

            num >= discard_alignment &&

            sector_num % discard_alignment) {

            if (num > discard_alignment) {

                num = discard_alignment;

            }

            num -= sector_num % discard_alignment;

        }



        /* limit request size */

        if (num > max_discard) {

            num = max_discard;

        }



        if (bs->drv->bdrv_co_discard) {

            ret = bs->drv->bdrv_co_discard(bs, sector_num, num);

        } else {

            BlockAIOCB *acb;

            CoroutineIOCompletion co = {

                .coroutine = qemu_coroutine_self(),

            };



            acb = bs->drv->bdrv_aio_discard(bs, sector_num, nb_sectors,

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



        sector_num += num;

        nb_sectors -= num;

    }

    ret = 0;

out:


    bdrv_set_dirty(bs, req.offset >> BDRV_SECTOR_BITS,

                   req.bytes >> BDRV_SECTOR_BITS);

    tracked_request_end(&req);

    return ret;

}