static int calculate_refcounts(BlockDriverState *bs, BdrvCheckResult *res,
                               BdrvCheckMode fix, bool *rebuild,
                               void **refcount_table, int64_t *nb_clusters)
{
    BDRVQcow2State *s = bs->opaque;
    int64_t i;
    QCowSnapshot *sn;
    int ret;
    if (!*refcount_table) {
        int64_t old_size = 0;
        ret = realloc_refcount_array(s, refcount_table,
                                     &old_size, *nb_clusters);
            res->check_errors++;
    /* header */
                        0, s->cluster_size);
    /* current L1 table */
    ret = check_refcounts_l1(bs, res, refcount_table, nb_clusters,
                             s->l1_table_offset, s->l1_size, CHECK_FRAG_INFO);
    /* snapshots */
    for (i = 0; i < s->nb_snapshots; i++) {
        sn = s->snapshots + i;
        ret = check_refcounts_l1(bs, res, refcount_table, nb_clusters,
                                 sn->l1_table_offset, sn->l1_size, 0);
                        s->snapshots_offset, s->snapshots_size);
    /* refcount data */
                        s->refcount_table_offset,
                        s->refcount_table_size * sizeof(uint64_t));
    return check_refblocks(bs, res, fix, rebuild, refcount_table, nb_clusters);