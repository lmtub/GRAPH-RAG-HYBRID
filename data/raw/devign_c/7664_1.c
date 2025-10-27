static int check_refcounts_l1(BlockDriverState *bs,

                              BdrvCheckResult *res,

                              uint16_t *refcount_table,

                              int refcount_table_size,

                              int64_t l1_table_offset, int l1_size,

                              int flags)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l1_table, l2_offset, l1_size2;

    int i, ret;



    l1_size2 = l1_size * sizeof(uint64_t);



    /* Mark L1 table as used */

    inc_refcounts(bs, res, refcount_table, refcount_table_size,

        l1_table_offset, l1_size2);



    /* Read L1 table entries from disk */

    if (l1_size2 == 0) {

        l1_table = NULL;

    } else {

        l1_table = g_malloc(l1_size2);

        if (bdrv_pread(bs->file, l1_table_offset,

                       l1_table, l1_size2) != l1_size2)

            goto fail;

        for(i = 0;i < l1_size; i++)

            be64_to_cpus(&l1_table[i]);

    }



    /* Do the actual checks */

    for(i = 0; i < l1_size; i++) {

        l2_offset = l1_table[i];

        if (l2_offset) {

            /* Mark L2 table as used */

            l2_offset &= L1E_OFFSET_MASK;

            inc_refcounts(bs, res, refcount_table, refcount_table_size,

                l2_offset, s->cluster_size);



            /* L2 tables are cluster aligned */

            if (offset_into_cluster(s, l2_offset)) {

                fprintf(stderr, "ERROR l2_offset=%" PRIx64 ": Table is not "

                    "cluster aligned; L1 entry corrupted\n", l2_offset);

                res->corruptions++;

            }



            /* Process and check L2 entries */

            ret = check_refcounts_l2(bs, res, refcount_table,

                                     refcount_table_size, l2_offset, flags);

            if (ret < 0) {

                goto fail;

            }

        }

    }

    g_free(l1_table);

    return 0;



fail:

    fprintf(stderr, "ERROR: I/O error in check_refcounts_l1\n");

    res->check_errors++;

    g_free(l1_table);

    return -EIO;

}
