static int inc_refcounts(BlockDriverState *bs,

                          uint16_t *refcount_table,

                          int refcount_table_size,

                          int64_t offset, int64_t size)

{

    BDRVQcowState *s = bs->opaque;

    int64_t start, last, cluster_offset;

    int k;

    int errors = 0;



    if (size <= 0)

        return 0;



    start = offset & ~(s->cluster_size - 1);

    last = (offset + size - 1) & ~(s->cluster_size - 1);

    for(cluster_offset = start; cluster_offset <= last;

        cluster_offset += s->cluster_size) {

        k = cluster_offset >> s->cluster_bits;

        if (k < 0 || k >= refcount_table_size) {

            fprintf(stderr, "ERROR: invalid cluster offset=0x%" PRIx64 "\n",

                cluster_offset);

            errors++;

        } else {

            if (++refcount_table[k] == 0) {

                fprintf(stderr, "ERROR: overflow cluster offset=0x%" PRIx64

                    "\n", cluster_offset);

                errors++;

            }

        }

    }



    return errors;

}
