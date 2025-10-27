void bdrv_drained_begin(BlockDriverState *bs)

{

    if (!bs->quiesce_counter++) {

        aio_disable_external(bdrv_get_aio_context(bs));

        bdrv_parent_drained_begin(bs);

    }



    bdrv_io_unplugged_begin(bs);

    bdrv_drain_recurse(bs);

    if (qemu_in_coroutine()) {

        bdrv_co_yield_to_drain(bs);

    } else {

        bdrv_drain_poll(bs);

    }

    bdrv_io_unplugged_end(bs);

}
