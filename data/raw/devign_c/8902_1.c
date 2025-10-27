static int check_refblocks(BlockDriverState *bs, BdrvCheckResult *res,

                           BdrvCheckMode fix, uint16_t **refcount_table,

                           int64_t *nb_clusters)

{

    BDRVQcowState *s = bs->opaque;

    int64_t i, size;

    int ret;



    for(i = 0; i < s->refcount_table_size; i++) {

        uint64_t offset, cluster;

        offset = s->refcount_table[i];

        cluster = offset >> s->cluster_bits;



        /* Refcount blocks are cluster aligned */

        if (offset_into_cluster(s, offset)) {

            fprintf(stderr, "ERROR refcount block %" PRId64 " is not "

                "cluster aligned; refcount table entry corrupted\n", i);

            res->corruptions++;

            continue;

        }



        if (cluster >= *nb_clusters) {

            fprintf(stderr, "%s refcount block %" PRId64 " is outside image\n",

                    fix & BDRV_FIX_ERRORS ? "Repairing" : "ERROR", i);



            if (fix & BDRV_FIX_ERRORS) {

                int64_t old_nb_clusters = *nb_clusters;

                uint16_t *new_refcount_table;



                if (offset > INT64_MAX - s->cluster_size) {

                    ret = -EINVAL;

                    goto resize_fail;

                }



                ret = bdrv_truncate(bs->file, offset + s->cluster_size);

                if (ret < 0) {

                    goto resize_fail;

                }

                size = bdrv_getlength(bs->file);

                if (size < 0) {

                    ret = size;

                    goto resize_fail;

                }



                *nb_clusters = size_to_clusters(s, size);

                assert(*nb_clusters >= old_nb_clusters);



                new_refcount_table = g_try_realloc(*refcount_table,

                                                   *nb_clusters *

                                                   sizeof(**refcount_table));

                if (!new_refcount_table) {

                    *nb_clusters = old_nb_clusters;

                    res->check_errors++;

                    return -ENOMEM;

                }

                *refcount_table = new_refcount_table;



                memset(*refcount_table + old_nb_clusters, 0,

                       (*nb_clusters - old_nb_clusters) *

                       sizeof(**refcount_table));



                if (cluster >= *nb_clusters) {

                    ret = -EINVAL;

                    goto resize_fail;

                }



                res->corruptions_fixed++;

                ret = inc_refcounts(bs, res, refcount_table, nb_clusters,

                                    offset, s->cluster_size);

                if (ret < 0) {

                    return ret;

                }

                /* No need to check whether the refcount is now greater than 1:

                 * This area was just allocated and zeroed, so it can only be

                 * exactly 1 after inc_refcounts() */

                continue;



resize_fail:

                res->corruptions++;

                fprintf(stderr, "ERROR could not resize image: %s\n",

                        strerror(-ret));

            } else {

                res->corruptions++;

            }

            continue;

        }



        if (offset != 0) {

            ret = inc_refcounts(bs, res, refcount_table, nb_clusters,

                                offset, s->cluster_size);

            if (ret < 0) {

                return ret;

            }

            if ((*refcount_table)[cluster] != 1) {

                fprintf(stderr, "%s refcount block %" PRId64

                    " refcount=%d\n",

                    fix & BDRV_FIX_ERRORS ? "Repairing" :

                                            "ERROR",

                    i, (*refcount_table)[cluster]);



                if (fix & BDRV_FIX_ERRORS) {

                    int64_t new_offset;



                    new_offset = realloc_refcount_block(bs, i, offset);

                    if (new_offset < 0) {

                        res->corruptions++;

                        continue;

                    }



                    /* update refcounts */

                    if ((new_offset >> s->cluster_bits) >= *nb_clusters) {

                        /* increase refcount_table size if necessary */

                        int old_nb_clusters = *nb_clusters;

                        *nb_clusters = (new_offset >> s->cluster_bits) + 1;

                        *refcount_table = g_renew(uint16_t, *refcount_table,

                                                  *nb_clusters);

                        memset(&(*refcount_table)[old_nb_clusters], 0,

                               (*nb_clusters - old_nb_clusters) *

                               sizeof(**refcount_table));

                    }

                    (*refcount_table)[cluster]--;

                    ret = inc_refcounts(bs, res, refcount_table, nb_clusters,

                                        new_offset, s->cluster_size);

                    if (ret < 0) {

                        return ret;

                    }



                    res->corruptions_fixed++;

                } else {

                    res->corruptions++;

                }

            }

        }

    }



    return 0;

}
