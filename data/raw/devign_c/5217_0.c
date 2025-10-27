static int check_refcounts_l2(BlockDriverState *bs, BdrvCheckResult *res,

    uint16_t *refcount_table, int64_t refcount_table_size, int64_t l2_offset,

    int flags)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l2_table, l2_entry;

    uint64_t next_contiguous_offset = 0;

    int i, l2_size, nb_csectors, ret;



    /* Read L2 table from disk */

    l2_size = s->l2_size * sizeof(uint64_t);

    l2_table = g_malloc(l2_size);



    ret = bdrv_pread(bs->file, l2_offset, l2_table, l2_size);

    if (ret < 0) {

        fprintf(stderr, "ERROR: I/O error in check_refcounts_l2\n");

        res->check_errors++;

        goto fail;

    }



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

            ret = inc_refcounts(bs, res, refcount_table, refcount_table_size,

                                l2_entry & ~511, nb_csectors * 512);

            if (ret < 0) {

                goto fail;

            }



            if (flags & CHECK_FRAG_INFO) {

                res->bfi.allocated_clusters++;

                res->bfi.compressed_clusters++;



                /* Compressed clusters are fragmented by nature.  Since they

                 * take up sub-sector space but we only have sector granularity

                 * I/O we need to re-read the same sectors even for adjacent

                 * compressed clusters.

                 */

                res->bfi.fragmented_clusters++;

            }

            break;



        case QCOW2_CLUSTER_ZERO:

            if ((l2_entry & L2E_OFFSET_MASK) == 0) {

                break;

            }

            /* fall through */



        case QCOW2_CLUSTER_NORMAL:

        {

            uint64_t offset = l2_entry & L2E_OFFSET_MASK;



            if (flags & CHECK_FRAG_INFO) {

                res->bfi.allocated_clusters++;

                if (next_contiguous_offset &&

                    offset != next_contiguous_offset) {

                    res->bfi.fragmented_clusters++;

                }

                next_contiguous_offset = offset + s->cluster_size;

            }



            /* Mark cluster as used */

            ret = inc_refcounts(bs, res, refcount_table, refcount_table_size,

                                offset, s->cluster_size);

            if (ret < 0) {

                goto fail;

            }



            /* Correct offsets are cluster aligned */

            if (offset_into_cluster(s, offset)) {

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

    g_free(l2_table);

    return ret;

}
