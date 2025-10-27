void bdrv_detach_aio_context(BlockDriverState *bs)

{

    BdrvAioNotifier *baf;



    if (!bs->drv) {

        return;

    }



    QLIST_FOREACH(baf, &bs->aio_notifiers, list) {

        baf->detach_aio_context(baf->opaque);

    }



    if (bs->io_limits_enabled) {

        throttle_timers_detach_aio_context(&bs->throttle_timers);

    }

    if (bs->drv->bdrv_detach_aio_context) {

        bs->drv->bdrv_detach_aio_context(bs);

    }

    if (bs->file) {

        bdrv_detach_aio_context(bs->file->bs);

    }

    if (bs->backing) {

        bdrv_detach_aio_context(bs->backing->bs);

    }



    bs->aio_context = NULL;

}
