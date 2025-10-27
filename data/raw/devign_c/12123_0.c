int qcow2_check_refcounts(BlockDriverState *bs, BdrvCheckResult *res,

                          BdrvCheckMode fix)

{

    BDRVQcowState *s = bs->opaque;

    int64_t size, i, highest_cluster;

    int nb_clusters, refcount1, refcount2;

    QCowSnapshot *sn;

    uint16_t *refcount_table;

    int ret;



    size = bdrv_getlength(bs->file);

    nb_clusters = size_to_clusters(s, size);

    refcount_table = g_malloc0(nb_clusters * sizeof(uint16_t));



    /* header */

    inc_refcounts(bs, res, refcount_table, nb_clusters,

        0, s->cluster_size);



    /* current L1 table */

    ret = check_refcounts_l1(bs, res, refcount_table, nb_clusters,

                       s->l1_table_offset, s->l1_size, 1);

    if (ret < 0) {

        goto fail;

    }



    /* snapshots */

    for(i = 0; i < s->nb_snapshots; i++) {

        sn = s->snapshots + i;

        ret = check_refcounts_l1(bs, res, refcount_table, nb_clusters,

            sn->l1_table_offset, sn->l1_size, 0);

        if (ret < 0) {

            goto fail;

        }

    }

    inc_refcounts(bs, res, refcount_table, nb_clusters,

        s->snapshots_offset, s->snapshots_size);



    /* refcount data */

    inc_refcounts(bs, res, refcount_table, nb_clusters,

        s->refcount_table_offset,

        s->refcount_table_size * sizeof(uint64_t));



    for(i = 0; i < s->refcount_table_size; i++) {

        uint64_t offset, cluster;

        offset = s->refcount_table[i];

        cluster = offset >> s->cluster_bits;



        /* Refcount blocks are cluster aligned */

        if (offset & (s->cluster_size - 1)) {

            fprintf(stderr, "ERROR refcount block %" PRId64 " is not "

                "cluster aligned; refcount table entry corrupted\n", i);

            res->corruptions++;

            continue;

        }



        if (cluster >= nb_clusters) {

            fprintf(stderr, "ERROR refcount block %" PRId64

                    " is outside image\n", i);

            res->corruptions++;

            continue;

        }



        if (offset != 0) {

            inc_refcounts(bs, res, refcount_table, nb_clusters,

                offset, s->cluster_size);

            if (refcount_table[cluster] != 1) {

                fprintf(stderr, "ERROR refcount block %" PRId64

                    " refcount=%d\n",

                    i, refcount_table[cluster]);

                res->corruptions++;

            }

        }

    }



    /* compare ref counts */

    for (i = 0, highest_cluster = 0; i < nb_clusters; i++) {

        refcount1 = get_refcount(bs, i);

        if (refcount1 < 0) {

            fprintf(stderr, "Can't get refcount for cluster %" PRId64 ": %s\n",

                i, strerror(-refcount1));

            res->check_errors++;

            continue;

        }



        refcount2 = refcount_table[i];



        if (refcount1 > 0 || refcount2 > 0) {

            highest_cluster = i;

        }



        if (refcount1 != refcount2) {



            /* Check if we're allowed to fix the mismatch */

            int *num_fixed = NULL;

            if (refcount1 > refcount2 && (fix & BDRV_FIX_LEAKS)) {

                num_fixed = &res->leaks_fixed;

            } else if (refcount1 < refcount2 && (fix & BDRV_FIX_ERRORS)) {

                num_fixed = &res->corruptions_fixed;

            }



            fprintf(stderr, "%s cluster %" PRId64 " refcount=%d reference=%d\n",

                   num_fixed != NULL     ? "Repairing" :

                   refcount1 < refcount2 ? "ERROR" :

                                           "Leaked",

                   i, refcount1, refcount2);



            if (num_fixed) {

                ret = update_refcount(bs, i << s->cluster_bits, 1,

                                      refcount2 - refcount1);

                if (ret >= 0) {

                    (*num_fixed)++;

                    continue;

                }

            }



            /* And if we couldn't, print an error */

            if (refcount1 < refcount2) {

                res->corruptions++;

            } else {

                res->leaks++;

            }

        }

    }



    res->image_end_offset = (highest_cluster + 1) * s->cluster_size;

    ret = 0;



fail:

    g_free(refcount_table);



    return ret;

}
