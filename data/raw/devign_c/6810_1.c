uint64_t qcow2_alloc_compressed_cluster_offset(BlockDriverState *bs,

                                               uint64_t offset,

                                               int compressed_size)

{

    BDRVQcowState *s = bs->opaque;

    int l2_index, ret;

    uint64_t *l2_table;

    int64_t cluster_offset;

    int nb_csectors;



    ret = get_cluster_table(bs, offset, &l2_table, &l2_index);

    if (ret < 0) {

        return 0;

    }



    cluster_offset = be64_to_cpu(l2_table[l2_index]);

    if (cluster_offset & QCOW_OFLAG_COPIED) {

        qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

        return 0;

    }



    if (cluster_offset)

        qcow2_free_any_clusters(bs, cluster_offset, 1);



    cluster_offset = qcow2_alloc_bytes(bs, compressed_size);

    if (cluster_offset < 0) {

        qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

        return 0;

    }



    nb_csectors = ((cluster_offset + compressed_size - 1) >> 9) -

                  (cluster_offset >> 9);



    cluster_offset |= QCOW_OFLAG_COMPRESSED |

                      ((uint64_t)nb_csectors << s->csize_shift);



    /* update L2 table */



    /* compressed clusters never have the copied flag */



    BLKDBG_EVENT(bs->file, BLKDBG_L2_UPDATE_COMPRESSED);

    qcow2_cache_entry_mark_dirty(s->l2_table_cache, l2_table);

    l2_table[l2_index] = cpu_to_be64(cluster_offset);

    ret = qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

    if (ret < 0) {

        return 0;

    }



    return cluster_offset;

}
