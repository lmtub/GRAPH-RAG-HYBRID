static bool is_zero_cluster_top_locked(BlockDriverState *bs, int64_t start)

{

    BDRVQcow2State *s = bs->opaque;

    int nr = s->cluster_sectors;

    uint64_t off;

    int ret;



    ret = qcow2_get_cluster_offset(bs, start << BDRV_SECTOR_BITS, &nr, &off);

    assert(nr == s->cluster_sectors);

    return ret == QCOW2_CLUSTER_UNALLOCATED || ret == QCOW2_CLUSTER_ZERO;

}
