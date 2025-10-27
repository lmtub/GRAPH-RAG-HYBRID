static int coroutine_fn raw_co_preadv(BlockDriverState *bs, uint64_t offset,

                                      uint64_t bytes, QEMUIOVector *qiov,

                                      int flags)

{

    BDRVRawState *s = bs->opaque;



    if (offset > UINT64_MAX - s->offset) {

        return -EINVAL;

    }

    offset += s->offset;



    BLKDBG_EVENT(bs->file, BLKDBG_READ_AIO);

    return bdrv_co_preadv(bs->file, offset, bytes, qiov, flags);

}
