static bool bdrv_requests_pending_all(void)

{

    BlockDriverState *bs;

    QTAILQ_FOREACH(bs, &bdrv_states, device_list) {

        if (bdrv_requests_pending(bs)) {

            return true;

        }

    }

    return false;

}
