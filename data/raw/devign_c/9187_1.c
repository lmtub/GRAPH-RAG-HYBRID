static void compare_refcounts(BlockDriverState *bs, BdrvCheckResult *res,

                              BdrvCheckMode fix, int64_t *highest_cluster,

                              uint16_t *refcount_table, int64_t nb_clusters)

{

    BDRVQcowState *s = bs->opaque;

    int64_t i;

    int refcount1, refcount2, ret;



    for (i = 0, *highest_cluster = 0; i < nb_clusters; i++) {

        refcount1 = get_refcount(bs, i);

        if (refcount1 < 0) {

            fprintf(stderr, "Can't get refcount for cluster %" PRId64 ": %s\n",

                i, strerror(-refcount1));

            res->check_errors++;

            continue;

        }



        refcount2 = refcount_table[i];



        if (refcount1 > 0 || refcount2 > 0) {

            *highest_cluster = i;

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

                                      refcount2 - refcount1,

                                      QCOW2_DISCARD_ALWAYS);

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

}
