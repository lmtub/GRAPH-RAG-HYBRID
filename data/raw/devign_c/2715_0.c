static int do_co_pwrite_zeroes(BlockBackend *blk, int64_t offset,

                               int64_t count, int flags, int64_t *total)

{

    Coroutine *co;

    CoWriteZeroes data = {

        .blk    = blk,

        .offset = offset,

        .count  = count,

        .total  = total,

        .flags  = flags,

        .done   = false,

    };



    if (count >> BDRV_SECTOR_BITS > INT_MAX) {

        return -ERANGE;

    }



    co = qemu_coroutine_create(co_pwrite_zeroes_entry, &data);

    qemu_coroutine_enter(co);

    while (!data.done) {

        aio_poll(blk_get_aio_context(blk), true);

    }

    if (data.ret < 0) {

        return data.ret;

    } else {

        return 1;

    }

}
