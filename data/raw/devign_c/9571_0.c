static void bdrv_throttle_read_timer_cb(void *opaque)

{

    BlockDriverState *bs = opaque;

    qemu_co_enter_next(&bs->throttled_reqs[0]);

}
