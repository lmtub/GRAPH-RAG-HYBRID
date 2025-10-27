static int check_refcounts_l2(BlockDriverState *bs,

    uint16_t *refcount_table, int refcount_table_size, int64_t l2_offset,

    int check_copied)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l2_table, offset;

    int i, l2_size, nb_csectors, refcount;

    int errors = 0;



    /* Read L2 table from disk */

    l2_size = s->l2_size * sizeof(uint64_t);

    l2_table = qemu_malloc(l2_size);



    if (bdrv_pread(bs->file, l2_offset, l2_table, l2_size) != l2_size)

        goto fail;



    /* Do the actual checks */

    for(i = 0; i < s->l2_size; i++) {

        offset = be64_to_cpu(l2_table[i]);

        if (offset != 0) {

            if (offset & QCOW_OFLAG_COMPRESSED) {

                /* Compressed clusters don't have QCOW_OFLAG_COPIED */

                if (offset & QCOW_OFLAG_COPIED) {

                    fprintf(stderr, "ERROR: cluster %" PRId64 ": "

                        "copied flag must never be set for compressed "

                        "clusters\n", offset >> s->cluster_bits);

                    offset &= ~QCOW_OFLAG_COPIED;

                    errors++;

                }



                /* Mark cluster as used */

                nb_csectors = ((offset >> s->csize_shift) &

                               s->csize_mask) + 1;

                offset &= s->cluster_offset_mask;

                errors += inc_refcounts(bs, refcount_table,

                              refcount_table_size,

                              offset & ~511, nb_csectors * 512);

            } else {

                /* QCOW_OFLAG_COPIED must be set iff refcount == 1 */

                if (check_copied) {

                    uint64_t entry = offset;

                    offset &= ~QCOW_OFLAG_COPIED;

                    refcount = get_refcount(bs, offset >> s->cluster_bits);

                    if (refcount < 0) {

                        fprintf(stderr, "Can't get refcount for offset %"

                            PRIx64 ": %s\n", entry, strerror(-refcount));

                    }

                    if ((refcount == 1) != ((entry & QCOW_OFLAG_COPIED) != 0)) {

                        fprintf(stderr, "ERROR OFLAG_COPIED: offset=%"

                            PRIx64 " refcount=%d\n", entry, refcount);

                        errors++;

                    }

                }



                /* Mark cluster as used */

                offset &= ~QCOW_OFLAG_COPIED;

                errors += inc_refcounts(bs, refcount_table,

                              refcount_table_size,

                              offset, s->cluster_size);



                /* Correct offsets are cluster aligned */

                if (offset & (s->cluster_size - 1)) {

                    fprintf(stderr, "ERROR offset=%" PRIx64 ": Cluster is not "

                        "properly aligned; L2 entry corrupted.\n", offset);

                    errors++;

                }

            }

        }

    }



    qemu_free(l2_table);

    return errors;



fail:

    fprintf(stderr, "ERROR: I/O error in check_refcounts_l1\n");

    qemu_free(l2_table);

    return -EIO;

}
