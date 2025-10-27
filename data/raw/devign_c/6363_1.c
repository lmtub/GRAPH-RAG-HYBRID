static int inc_refcounts(BlockDriverState *bs,

                         BdrvCheckResult *res,

                         uint16_t **refcount_table,

                         int64_t *refcount_table_size,

                         int64_t offset, int64_t size)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t start, last, cluster_offset, k;

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



        if (++(*refcount_table)[k] == 0) {

            fprintf(stderr, "ERROR: overflow cluster offset=0x%" PRIx64

                    "\n", cluster_offset);

            res->corruptions++;

        }

    }



    return 0;

}
