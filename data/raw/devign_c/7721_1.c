static int64_t coroutine_fn qcow_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum, BlockDriverState **file)

{

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster, n;

    uint64_t cluster_offset;



    qemu_co_mutex_lock(&s->lock);

    cluster_offset = get_cluster_offset(bs, sector_num << 9, 0, 0, 0, 0);

    qemu_co_mutex_unlock(&s->lock);

    index_in_cluster = sector_num & (s->cluster_sectors - 1);

    n = s->cluster_sectors - index_in_cluster;

    if (n > nb_sectors)

        n = nb_sectors;

    *pnum = n;

    if (!cluster_offset) {

        return 0;

    }

    if ((cluster_offset & QCOW_OFLAG_COMPRESSED) || s->crypto) {

        return BDRV_BLOCK_DATA;

    }

    cluster_offset |= (index_in_cluster << BDRV_SECTOR_BITS);

    *file = bs->file->bs;

    return BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | cluster_offset;

}
