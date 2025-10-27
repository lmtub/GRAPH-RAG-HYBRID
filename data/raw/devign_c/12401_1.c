static int vmdk_is_allocated(BlockDriverState *bs, int64_t sector_num, 

                             int nb_sectors, int *pnum)

{

    BDRVVmdkState *s = bs->opaque;

    int index_in_cluster, n;

    uint64_t cluster_offset;



    cluster_offset = get_cluster_offset(bs, sector_num << 9, 0);

    index_in_cluster = sector_num % s->cluster_sectors;

    n = s->cluster_sectors - index_in_cluster;

    if (n > nb_sectors)

        n = nb_sectors;

    *pnum = n;

    return (cluster_offset != 0);

}
