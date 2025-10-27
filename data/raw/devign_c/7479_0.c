static int expand_zero_clusters_in_l1(BlockDriverState *bs, uint64_t *l1_table,

                                      int l1_size, int64_t *visited_l1_entries,

                                      int64_t l1_entries,

                                      BlockDriverAmendStatusCB *status_cb,

                                      void *cb_opaque)

{

    BDRVQcow2State *s = bs->opaque;

    bool is_active_l1 = (l1_table == s->l1_table);

    uint64_t *l2_table = NULL;

    int ret;

    int i, j;



    if (!is_active_l1) {

        /* inactive L2 tables require a buffer to be stored in when loading

         * them from disk */

        l2_table = qemu_try_blockalign(bs->file->bs, s->cluster_size);

        if (l2_table == NULL) {

            return -ENOMEM;

        }

    }



    for (i = 0; i < l1_size; i++) {

        uint64_t l2_offset = l1_table[i] & L1E_OFFSET_MASK;

        bool l2_dirty = false;

        uint64_t l2_refcount;



        if (!l2_offset) {

            /* unallocated */

            (*visited_l1_entries)++;

            if (status_cb) {

                status_cb(bs, *visited_l1_entries, l1_entries, cb_opaque);

            }

            continue;

        }



        if (offset_into_cluster(s, l2_offset)) {

            qcow2_signal_corruption(bs, true, -1, -1, "L2 table offset %#"

                                    PRIx64 " unaligned (L1 index: %#x)",

                                    l2_offset, i);

            ret = -EIO;

            goto fail;

        }



        if (is_active_l1) {

            /* get active L2 tables from cache */

            ret = qcow2_cache_get(bs, s->l2_table_cache, l2_offset,

                    (void **)&l2_table);

        } else {

            /* load inactive L2 tables from disk */

            ret = bdrv_read(bs->file, l2_offset / BDRV_SECTOR_SIZE,

                            (void *)l2_table, s->cluster_sectors);

        }

        if (ret < 0) {

            goto fail;

        }



        ret = qcow2_get_refcount(bs, l2_offset >> s->cluster_bits,

                                 &l2_refcount);

        if (ret < 0) {

            goto fail;

        }



        for (j = 0; j < s->l2_size; j++) {

            uint64_t l2_entry = be64_to_cpu(l2_table[j]);

            int64_t offset = l2_entry & L2E_OFFSET_MASK;

            QCow2ClusterType cluster_type = qcow2_get_cluster_type(l2_entry);

            bool preallocated = offset != 0;



            if (cluster_type != QCOW2_CLUSTER_ZERO) {

                continue;

            }



            if (!preallocated) {

                if (!bs->backing) {

                    /* not backed; therefore we can simply deallocate the

                     * cluster */

                    l2_table[j] = 0;

                    l2_dirty = true;

                    continue;

                }



                offset = qcow2_alloc_clusters(bs, s->cluster_size);

                if (offset < 0) {

                    ret = offset;

                    goto fail;

                }



                if (l2_refcount > 1) {

                    /* For shared L2 tables, set the refcount accordingly (it is

                     * already 1 and needs to be l2_refcount) */

                    ret = qcow2_update_cluster_refcount(bs,

                            offset >> s->cluster_bits,

                            refcount_diff(1, l2_refcount), false,

                            QCOW2_DISCARD_OTHER);

                    if (ret < 0) {

                        qcow2_free_clusters(bs, offset, s->cluster_size,

                                            QCOW2_DISCARD_OTHER);

                        goto fail;

                    }

                }

            }



            if (offset_into_cluster(s, offset)) {

                qcow2_signal_corruption(bs, true, -1, -1, "Data cluster offset "

                                        "%#" PRIx64 " unaligned (L2 offset: %#"

                                        PRIx64 ", L2 index: %#x)", offset,

                                        l2_offset, j);

                if (!preallocated) {

                    qcow2_free_clusters(bs, offset, s->cluster_size,

                                        QCOW2_DISCARD_ALWAYS);

                }

                ret = -EIO;

                goto fail;

            }



            ret = qcow2_pre_write_overlap_check(bs, 0, offset, s->cluster_size);

            if (ret < 0) {

                if (!preallocated) {

                    qcow2_free_clusters(bs, offset, s->cluster_size,

                                        QCOW2_DISCARD_ALWAYS);

                }

                goto fail;

            }



            ret = bdrv_pwrite_zeroes(bs->file, offset, s->cluster_size, 0);

            if (ret < 0) {

                if (!preallocated) {

                    qcow2_free_clusters(bs, offset, s->cluster_size,

                                        QCOW2_DISCARD_ALWAYS);

                }

                goto fail;

            }



            if (l2_refcount == 1) {

                l2_table[j] = cpu_to_be64(offset | QCOW_OFLAG_COPIED);

            } else {

                l2_table[j] = cpu_to_be64(offset);

            }

            l2_dirty = true;

        }



        if (is_active_l1) {

            if (l2_dirty) {

                qcow2_cache_entry_mark_dirty(bs, s->l2_table_cache, l2_table);

                qcow2_cache_depends_on_flush(s->l2_table_cache);

            }

            qcow2_cache_put(bs, s->l2_table_cache, (void **) &l2_table);

        } else {

            if (l2_dirty) {

                ret = qcow2_pre_write_overlap_check(bs,

                        QCOW2_OL_INACTIVE_L2 | QCOW2_OL_ACTIVE_L2, l2_offset,

                        s->cluster_size);

                if (ret < 0) {

                    goto fail;

                }



                ret = bdrv_write(bs->file, l2_offset / BDRV_SECTOR_SIZE,

                                 (void *)l2_table, s->cluster_sectors);

                if (ret < 0) {

                    goto fail;

                }

            }

        }



        (*visited_l1_entries)++;

        if (status_cb) {

            status_cb(bs, *visited_l1_entries, l1_entries, cb_opaque);

        }

    }



    ret = 0;



fail:

    if (l2_table) {

        if (!is_active_l1) {

            qemu_vfree(l2_table);

        } else {

            qcow2_cache_put(bs, s->l2_table_cache, (void **) &l2_table);

        }

    }

    return ret;

}
