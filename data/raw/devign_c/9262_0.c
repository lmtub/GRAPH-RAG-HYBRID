static bool is_zero_cluster(BlockDriverState *bs, int64_t start)

{

    BDRVQcow2State *s = bs->opaque;

    int nr;

    BlockDriverState *file;

    int64_t res = bdrv_get_block_status_above(bs, NULL, start,

                                              s->cluster_sectors, &nr, &file);

    return res >= 0 && (res & BDRV_BLOCK_ZERO);

}
