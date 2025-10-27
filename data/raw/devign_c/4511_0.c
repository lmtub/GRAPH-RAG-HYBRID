static bool bdrv_requests_pending(BlockDriverState *bs)

{

    if (!QLIST_EMPTY(&bs->tracked_requests)) {

        return true;

    }

    if (!qemu_co_queue_empty(&bs->throttled_reqs[0])) {

        return true;

    }

    if (!qemu_co_queue_empty(&bs->throttled_reqs[1])) {

        return true;

    }

    if (bs->file && bdrv_requests_pending(bs->file)) {

        return true;

    }

    if (bs->backing_hd && bdrv_requests_pending(bs->backing_hd)) {

        return true;

    }

    return false;

}
