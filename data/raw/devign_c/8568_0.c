static void check_refcounts(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    int64_t size;

    int nb_clusters, refcount1, refcount2, i;

    QCowSnapshot *sn;

    uint16_t *refcount_table;



    size = bdrv_getlength(s->hd);

    nb_clusters = size_to_clusters(s, size);

    refcount_table = qemu_mallocz(nb_clusters * sizeof(uint16_t));



    /* header */

    inc_refcounts(bs, refcount_table, nb_clusters,

                  0, s->cluster_size);



    check_refcounts_l1(bs, refcount_table, nb_clusters,

                       s->l1_table_offset, s->l1_size, 1);



    /* snapshots */

    for(i = 0; i < s->nb_snapshots; i++) {

        sn = s->snapshots + i;

        check_refcounts_l1(bs, refcount_table, nb_clusters,

                           sn->l1_table_offset, sn->l1_size, 0);

    }

    inc_refcounts(bs, refcount_table, nb_clusters,

                  s->snapshots_offset, s->snapshots_size);



    /* refcount data */

    inc_refcounts(bs, refcount_table, nb_clusters,

                  s->refcount_table_offset,

                  s->refcount_table_size * sizeof(uint64_t));

    for(i = 0; i < s->refcount_table_size; i++) {

        int64_t offset;

        offset = s->refcount_table[i];

        if (offset != 0) {

            inc_refcounts(bs, refcount_table, nb_clusters,

                          offset, s->cluster_size);

        }

    }



    /* compare ref counts */

    for(i = 0; i < nb_clusters; i++) {

        refcount1 = get_refcount(bs, i);

        refcount2 = refcount_table[i];

        if (refcount1 != refcount2)

            fprintf(stderr, "ERROR cluster %d refcount=%d reference=%d\n",

                   i, refcount1, refcount2);

    }



    qemu_free(refcount_table);

}
