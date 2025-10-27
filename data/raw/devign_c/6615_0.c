int qcow2_update_snapshot_refcount(BlockDriverState *bs,

    int64_t l1_table_offset, int l1_size, int addend)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t *l1_table, *l2_table, l2_offset, offset, l1_size2, refcount;

    bool l1_allocated = false;

    int64_t old_offset, old_l2_offset;

    int i, j, l1_modified = 0, nb_csectors;

    int ret;



    assert(addend >= -1 && addend <= 1);



    l2_table = NULL;

    l1_table = NULL;

    l1_size2 = l1_size * sizeof(uint64_t);



    s->cache_discards = true;



    /* WARNING: qcow2_snapshot_goto relies on this function not using the

     * l1_table_offset when it is the current s->l1_table_offset! Be careful

     * when changing this! */

    if (l1_table_offset != s->l1_table_offset) {

        l1_table = g_try_malloc0(align_offset(l1_size2, 512));

        if (l1_size2 && l1_table == NULL) {

            ret = -ENOMEM;

            goto fail;

        }

        l1_allocated = true;



        ret = bdrv_pread(bs->file, l1_table_offset, l1_table, l1_size2);

        if (ret < 0) {

            goto fail;

        }



        for(i = 0;i < l1_size; i++)

            be64_to_cpus(&l1_table[i]);

    } else {

        assert(l1_size == s->l1_size);

        l1_table = s->l1_table;

        l1_allocated = false;

    }



    for(i = 0; i < l1_size; i++) {

        l2_offset = l1_table[i];

        if (l2_offset) {

            old_l2_offset = l2_offset;

            l2_offset &= L1E_OFFSET_MASK;



            if (offset_into_cluster(s, l2_offset)) {

                qcow2_signal_corruption(bs, true, -1, -1, "L2 table offset %#"

                                        PRIx64 " unaligned (L1 index: %#x)",

                                        l2_offset, i);

                ret = -EIO;

                goto fail;

            }



            ret = qcow2_cache_get(bs, s->l2_table_cache, l2_offset,

                (void**) &l2_table);

            if (ret < 0) {

                goto fail;

            }



            for(j = 0; j < s->l2_size; j++) {

                uint64_t cluster_index;



                offset = be64_to_cpu(l2_table[j]);

                old_offset = offset;

                offset &= ~QCOW_OFLAG_COPIED;



                switch (qcow2_get_cluster_type(offset)) {

                    case QCOW2_CLUSTER_COMPRESSED:

                        nb_csectors = ((offset >> s->csize_shift) &

                                       s->csize_mask) + 1;

                        if (addend != 0) {

                            ret = update_refcount(bs,

                                (offset & s->cluster_offset_mask) & ~511,

                                nb_csectors * 512, abs(addend), addend < 0,

                                QCOW2_DISCARD_SNAPSHOT);

                            if (ret < 0) {

                                goto fail;

                            }

                        }

                        /* compressed clusters are never modified */

                        refcount = 2;

                        break;



                    case QCOW2_CLUSTER_NORMAL:

                    case QCOW2_CLUSTER_ZERO:

                        if (offset_into_cluster(s, offset & L2E_OFFSET_MASK)) {

                            qcow2_signal_corruption(bs, true, -1, -1, "Data "

                                                    "cluster offset %#llx "

                                                    "unaligned (L2 offset: %#"

                                                    PRIx64 ", L2 index: %#x)",

                                                    offset & L2E_OFFSET_MASK,

                                                    l2_offset, j);

                            ret = -EIO;

                            goto fail;

                        }



                        cluster_index = (offset & L2E_OFFSET_MASK) >> s->cluster_bits;

                        if (!cluster_index) {

                            /* unallocated */

                            refcount = 0;

                            break;

                        }

                        if (addend != 0) {

                            ret = qcow2_update_cluster_refcount(bs,

                                    cluster_index, abs(addend), addend < 0,

                                    QCOW2_DISCARD_SNAPSHOT);

                            if (ret < 0) {

                                goto fail;

                            }

                        }



                        ret = qcow2_get_refcount(bs, cluster_index, &refcount);

                        if (ret < 0) {

                            goto fail;

                        }

                        break;



                    case QCOW2_CLUSTER_UNALLOCATED:

                        refcount = 0;

                        break;



                    default:

                        abort();

                }



                if (refcount == 1) {

                    offset |= QCOW_OFLAG_COPIED;

                }

                if (offset != old_offset) {

                    if (addend > 0) {

                        qcow2_cache_set_dependency(bs, s->l2_table_cache,

                            s->refcount_block_cache);

                    }

                    l2_table[j] = cpu_to_be64(offset);

                    qcow2_cache_entry_mark_dirty(bs, s->l2_table_cache,

                                                 l2_table);

                }

            }



            qcow2_cache_put(bs, s->l2_table_cache, (void **) &l2_table);



            if (addend != 0) {

                ret = qcow2_update_cluster_refcount(bs, l2_offset >>

                                                        s->cluster_bits,

                                                    abs(addend), addend < 0,

                                                    QCOW2_DISCARD_SNAPSHOT);

                if (ret < 0) {

                    goto fail;

                }

            }

            ret = qcow2_get_refcount(bs, l2_offset >> s->cluster_bits,

                                     &refcount);

            if (ret < 0) {

                goto fail;

            } else if (refcount == 1) {

                l2_offset |= QCOW_OFLAG_COPIED;

            }

            if (l2_offset != old_l2_offset) {

                l1_table[i] = l2_offset;

                l1_modified = 1;

            }

        }

    }



    ret = bdrv_flush(bs);

fail:

    if (l2_table) {

        qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

    }



    s->cache_discards = false;

    qcow2_process_discards(bs, ret);



    /* Update L1 only if it isn't deleted anyway (addend = -1) */

    if (ret == 0 && addend >= 0 && l1_modified) {

        for (i = 0; i < l1_size; i++) {

            cpu_to_be64s(&l1_table[i]);

        }



        ret = bdrv_pwrite_sync(bs->file, l1_table_offset,

                               l1_table, l1_size2);



        for (i = 0; i < l1_size; i++) {

            be64_to_cpus(&l1_table[i]);

        }

    }

    if (l1_allocated)

        g_free(l1_table);

    return ret;

}
