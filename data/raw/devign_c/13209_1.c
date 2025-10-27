static int inc_refcounts(BlockDriverState *bs,

                         BdrvCheckResult *res,

                         void **refcount_table,

                         int64_t *refcount_table_size,

                         int64_t offset, int64_t size)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t start, last, cluster_offset, k, refcount;

    int ret;



    if (size <= 0) {

        return 0;

    }



    start = start_of_cluster(s, offset);

    last = start_of_cluster(s, offset + size - 1);

    for(cluster_offset = start; cluster_offset <= last;

        cluster_offset += s->cluster_size) {

        k = cluster_offset >> s->cluster_bits;

        if (k >= *refcount_table_size) {

            ret = realloc_refcount_array(s, refcount_table,

                                         refcount_table_size, k + 1);

            if (ret < 0) {

                res->check_errors++;

                return ret;

            }

        }



        refcount = s->get_refcount(*refcount_table, k);

        if (refcount == s->refcount_max) {

            fprintf(stderr, "ERROR: overflow cluster offset=0x%" PRIx64

                    "\n", cluster_offset);




            res->corruptions++;

            continue;

        }

        s->set_refcount(*refcount_table, k, refcount + 1);

    }



    return 0;

}