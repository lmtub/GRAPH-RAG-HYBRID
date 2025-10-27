static int check_refcounts_l1(BlockDriverState *bs,

                              uint16_t *refcount_table,

                              int refcount_table_size,

                              int64_t l1_table_offset, int l1_size,

                              int check_copied)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l1_table, *l2_table, l2_offset, offset, l1_size2;

    int l2_size, i, j, nb_csectors, refcount;



    l2_table = NULL;

    l1_size2 = l1_size * sizeof(uint64_t);



    inc_refcounts(bs, refcount_table, refcount_table_size,

                  l1_table_offset, l1_size2);



    l1_table = qemu_malloc(l1_size2);

    if (bdrv_pread(s->hd, l1_table_offset,

                   l1_table, l1_size2) != l1_size2)

        goto fail;

    for(i = 0;i < l1_size; i++)

        be64_to_cpus(&l1_table[i]);



    l2_size = s->l2_size * sizeof(uint64_t);

    l2_table = qemu_malloc(l2_size);

    for(i = 0; i < l1_size; i++) {

        l2_offset = l1_table[i];

        if (l2_offset) {

            if (check_copied) {

                refcount = get_refcount(bs, (l2_offset & ~QCOW_OFLAG_COPIED) >> s->cluster_bits);

                if ((refcount == 1) != ((l2_offset & QCOW_OFLAG_COPIED) != 0)) {

                    printf("ERROR OFLAG_COPIED: l2_offset=%llx refcount=%d\n",

                           l2_offset, refcount);

                }

            }

            l2_offset &= ~QCOW_OFLAG_COPIED;

            if (bdrv_pread(s->hd, l2_offset, l2_table, l2_size) != l2_size)

                goto fail;

            for(j = 0; j < s->l2_size; j++) {

                offset = be64_to_cpu(l2_table[j]);

                if (offset != 0) {

                    if (offset & QCOW_OFLAG_COMPRESSED) {

                        if (offset & QCOW_OFLAG_COPIED) {

                            printf("ERROR: cluster %lld: copied flag must never be set for compressed clusters\n",

                                   offset >> s->cluster_bits);

                            offset &= ~QCOW_OFLAG_COPIED;

                        }

                        nb_csectors = ((offset >> s->csize_shift) &

                                       s->csize_mask) + 1;

                        offset &= s->cluster_offset_mask;

                        inc_refcounts(bs, refcount_table,

                                      refcount_table_size,

                                      offset & ~511, nb_csectors * 512);

                    } else {

                        if (check_copied) {

                            refcount = get_refcount(bs, (offset & ~QCOW_OFLAG_COPIED) >> s->cluster_bits);

                            if ((refcount == 1) != ((offset & QCOW_OFLAG_COPIED) != 0)) {

                                printf("ERROR OFLAG_COPIED: offset=%llx refcount=%d\n",

                                       offset, refcount);

                            }

                        }

                        offset &= ~QCOW_OFLAG_COPIED;

                        inc_refcounts(bs, refcount_table,

                                      refcount_table_size,

                                      offset, s->cluster_size);

                    }

                }

            }

            inc_refcounts(bs, refcount_table,

                          refcount_table_size,

                          l2_offset,

                          s->cluster_size);

        }

    }

    qemu_free(l1_table);

    qemu_free(l2_table);

    return 0;

 fail:

    printf("ERROR: I/O error in check_refcounts_l1\n");

    qemu_free(l1_table);

    qemu_free(l2_table);

    return -EIO;

}
