void bdrv_io_limits_enable(BlockDriverState *bs)

{

    qemu_co_queue_init(&bs->throttled_reqs);

    bs->block_timer = qemu_new_timer_ns(vm_clock, bdrv_block_timer, bs);

    bs->io_limits_enabled = true;

}
