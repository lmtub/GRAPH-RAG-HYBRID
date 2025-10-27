static int coroutine_fn blkreplay_co_flush(BlockDriverState *bs)

{

    uint64_t reqid = request_id++;

    int ret = bdrv_co_flush(bs->file->bs);

    block_request_create(reqid, bs, qemu_coroutine_self());

    qemu_coroutine_yield();



    return ret;

}
