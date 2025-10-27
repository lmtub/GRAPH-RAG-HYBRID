static int check_refcounts_l2(BlockDriverState *bs, BdrvCheckResult *res,

    uint16_t *refcount_table, int refcount_table_size, int64_t l2_offset,

    int check_copied)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l2_table, l2_entry;

    int i, l2_size, nb_csectors, refcount;



    /* Read L2 table from disk */

    l2_size = s->l2_size * sizeof(uint64_t);

    l2_table = g_malloc(l2_size);



    if (bdrv_pread(bs->file, l2_offset, l2_table, l2_size) != l2_size)

        goto fail;



    /* Do the actual checks */

    for(i = 0; i < s->l2_size; i++) {

        l2_entry = be64_to_cpu(l2_table[i]);



        switch (qcow2_get_cluster_type(l2_entry)) {

        case QCOW2_CLUSTER_COMPRESSED:

            /* Compressed clusters don't have QCOW_OFLAG_COPIED */

            if (l2_entry & QCOW_OFLAG_COPIED) {

                fprintf(stderr, "ERROR: cluster %" PRId64 ": "

                    "copied flag must never be set for compressed "

                    "clusters\n", l2_entry >> s->cluster_bits);

                l2_entry &= ~QCOW_OFLAG_COPIED;

                res->corruptions++;

            }



            /* Mark cluster as used */

            nb_csectors = ((l2_entry >> s->csize_shift) &

                           s->csize_mask) + 1;

            l2_entry &= s->cluster_offset_mask;

            inc_refcounts(bs, res, refcount_table, refcount_table_size,

                l2_entry & ~511, nb_csectors * 512);

            break;



        case QCOW2_CLUSTER_ZERO:

            if ((l2_entry & L2E_OFFSET_MASK) == 0) {

                break;

            }

            /* fall through */



        case QCOW2_CLUSTER_NORMAL:

        {

            /* QCOW_OFLAG_COPIED must be set iff refcount == 1 */

            uint64_t offset = l2_entry & L2E_OFFSET_MASK;



            if (check_copied) {

                refcount = get_refcount(bs, offset >> s->cluster_bits);

                if (refcount < 0) {

                    fprintf(stderr, "Can't get refcount for offset %"

                        PRIx64 ": %s\n", l2_entry, strerror(-refcount));

                    goto fail;

                }

                if ((refcount == 1) != ((l2_entry & QCOW_OFLAG_COPIED) != 0)) {

                    fprintf(stderr, "ERROR OFLAG_COPIED: offset=%"

                        PRIx64 " refcount=%d\n", l2_entry, refcount);

                    res->corruptions++;

                }

            }



            /* Mark cluster as used */

            inc_refcounts(bs, res, refcount_table,refcount_table_size,

                offset, s->cluster_size);



            /* Correct offsets are cluster aligned */

            if (offset & (s->cluster_size - 1)) {

                fprintf(stderr, "ERROR offset=%" PRIx64 ": Cluster is not "

                    "properly aligned; L2 entry corrupted.\n", offset);

                res->corruptions++;

            }

            break;

        }



        case QCOW2_CLUSTER_UNALLOCATED:

            break;



        default:

            abort();

        }

    }



    g_free(l2_table);

    return 0;



fail:

    fprintf(stderr, "ERROR: I/O error in check_refcounts_l2\n");

    g_free(l2_table);

    return -EIO;

}
