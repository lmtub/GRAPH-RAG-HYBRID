int qcow2_check_refcounts(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    int64_t size;

    int nb_clusters, refcount1, refcount2, i;

    QCowSnapshot *sn;

    uint16_t *refcount_table;

    int ret, errors = 0;



    size = bdrv_getlength(bs->file);

    nb_clusters = size_to_clusters(s, size);

    refcount_table = qemu_mallocz(nb_clusters * sizeof(uint16_t));



    /* header */

    errors += inc_refcounts(bs, refcount_table, nb_clusters,

                  0, s->cluster_size);



    /* current L1 table */

    ret = check_refcounts_l1(bs, refcount_table, nb_clusters,

                       s->l1_table_offset, s->l1_size, 1);

    if (ret < 0) {

        return ret;

    }

    errors += ret;



    /* snapshots */

    for(i = 0; i < s->nb_snapshots; i++) {

        sn = s->snapshots + i;

        check_refcounts_l1(bs, refcount_table, nb_clusters,

                           sn->l1_table_offset, sn->l1_size, 0);

    }

    errors += inc_refcounts(bs, refcount_table, nb_clusters,

                  s->snapshots_offset, s->snapshots_size);



    /* refcount data */

    errors += inc_refcounts(bs, refcount_table, nb_clusters,

                  s->refcount_table_offset,

                  s->refcount_table_size * sizeof(uint64_t));

    for(i = 0; i < s->refcount_table_size; i++) {

        int64_t offset;

        offset = s->refcount_table[i];



        /* Refcount blocks are cluster aligned */

        if (offset & (s->cluster_size - 1)) {

            fprintf(stderr, "ERROR refcount block %d is not "

                "cluster aligned; refcount table entry corrupted\n", i);

            errors++;

        }



        if (offset != 0) {

            errors += inc_refcounts(bs, refcount_table, nb_clusters,

                          offset, s->cluster_size);

            if (refcount_table[offset / s->cluster_size] != 1) {

                fprintf(stderr, "ERROR refcount block %d refcount=%d\n",

                    i, refcount_table[offset / s->cluster_size]);

            }

        }

    }



    /* compare ref counts */

    for(i = 0; i < nb_clusters; i++) {

        refcount1 = get_refcount(bs, i);

        if (refcount1 < 0) {

            fprintf(stderr, "Can't get refcount for cluster %d: %s\n",

                i, strerror(-refcount1));

        }



        refcount2 = refcount_table[i];

        if (refcount1 != refcount2) {

            fprintf(stderr, "ERROR cluster %d refcount=%d reference=%d\n",

                   i, refcount1, refcount2);

            errors++;

        }

    }



    qemu_free(refcount_table);



    return errors;

}
