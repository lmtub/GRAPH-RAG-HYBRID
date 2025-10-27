static int inc_refcounts(BlockDriverState *bs,

                         BdrvCheckResult *res,

                         uint16_t *refcount_table,

                         int64_t refcount_table_size,

                         int64_t offset, int64_t size)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t start, last, cluster_offset, k;



    if (size <= 0) {

        return 0;

    }



    start = start_of_cluster(s, offset);

    last = start_of_cluster(s, offset + size - 1);

    for(cluster_offset = start; cluster_offset <= last;

        cluster_offset += s->cluster_size) {

        k = cluster_offset >> s->cluster_bits;

        if (k >= refcount_table_size) {

            fprintf(stderr, "Warning: cluster offset=0x%" PRIx64 " is after "

                "the end of the image file, can't properly check refcounts.\n",

                cluster_offset);

            res->check_errors++;

        } else {

            if (++refcount_table[k] == 0) {

                fprintf(stderr, "ERROR: overflow cluster offset=0x%" PRIx64

                    "\n", cluster_offset);

                res->corruptions++;

            }

        }

    }



    return 0;

}
