static int coroutine_fn blkreplay_co_pwritev(BlockDriverState *bs,

    uint64_t offset, uint64_t bytes, QEMUIOVector *qiov, int flags)

{

    uint64_t reqid = request_id++;

    int ret = bdrv_co_pwritev(bs->file->bs, offset, bytes, qiov, flags);

    block_request_create(reqid, bs, qemu_coroutine_self());

    qemu_coroutine_yield();



    return ret;

}
