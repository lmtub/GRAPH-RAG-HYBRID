static bool is_zero_sectors(BlockDriverState *bs, int64_t start,
                            uint32_t count)
{
    int nr;
    BlockDriverState *file;
    int64_t res;
    if (!count) {
        return true;
    res = bdrv_get_block_status_above(bs, NULL, start, count,
                                      &nr, &file);
    return res >= 0 && (res & BDRV_BLOCK_ZERO) && nr == count;