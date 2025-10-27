static int coroutine_fn blkreplay_co_pwrite_zeroes(BlockDriverState *bs,

    int64_t offset, int count, BdrvRequestFlags flags)

{

    uint64_t reqid = request_id++;

    int ret = bdrv_co_pwrite_zeroes(bs->file, offset, count, flags);

    block_request_create(reqid, bs, qemu_coroutine_self());

    qemu_coroutine_yield();



    return ret;

}
