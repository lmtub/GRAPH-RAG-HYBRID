static int coroutine_fn raw_co_pwrite_zeroes(BlockDriverState *bs,

                                             int64_t offset, int count,

                                             BdrvRequestFlags flags)

{

    return bdrv_co_pwrite_zeroes(bs->file->bs, offset, count, flags);

}
