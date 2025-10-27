static int l2_allocate(BlockDriverState *bs, int l1_index, uint64_t **table)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t old_l2_offset;

    uint64_t *l2_table;

    int64_t l2_offset;

    int ret;



    old_l2_offset = s->l1_table[l1_index];



    trace_qcow2_l2_allocate(bs, l1_index);



    /* allocate a new l2 entry */



    l2_offset = qcow2_alloc_clusters(bs, s->l2_size * sizeof(uint64_t));

    if (l2_offset < 0) {

        return l2_offset;

    }



    ret = qcow2_cache_flush(bs, s->refcount_block_cache);

    if (ret < 0) {

        goto fail;

    }



    /* allocate a new entry in the l2 cache */



    trace_qcow2_l2_allocate_get_empty(bs, l1_index);

    ret = qcow2_cache_get_empty(bs, s->l2_table_cache, l2_offset, (void**) table);

    if (ret < 0) {

        return ret;

    }



    l2_table = *table;



    if ((old_l2_offset & L1E_OFFSET_MASK) == 0) {

        /* if there was no old l2 table, clear the new table */

        memset(l2_table, 0, s->l2_size * sizeof(uint64_t));

    } else {

        uint64_t* old_table;



        /* if there was an old l2 table, read it from the disk */

        BLKDBG_EVENT(bs->file, BLKDBG_L2_ALLOC_COW_READ);

        ret = qcow2_cache_get(bs, s->l2_table_cache,

            old_l2_offset & L1E_OFFSET_MASK,

            (void**) &old_table);

        if (ret < 0) {

            goto fail;

        }



        memcpy(l2_table, old_table, s->cluster_size);



        ret = qcow2_cache_put(bs, s->l2_table_cache, (void**) &old_table);

        if (ret < 0) {

            goto fail;

        }

    }



    /* write the l2 table to the file */

    BLKDBG_EVENT(bs->file, BLKDBG_L2_ALLOC_WRITE);



    trace_qcow2_l2_allocate_write_l2(bs, l1_index);

    qcow2_cache_entry_mark_dirty(s->l2_table_cache, l2_table);

    ret = qcow2_cache_flush(bs, s->l2_table_cache);

    if (ret < 0) {

        goto fail;

    }



    /* update the L1 entry */

    trace_qcow2_l2_allocate_write_l1(bs, l1_index);

    s->l1_table[l1_index] = l2_offset | QCOW_OFLAG_COPIED;

    ret = write_l1_entry(bs, l1_index);

    if (ret < 0) {

        goto fail;

    }



    *table = l2_table;

    trace_qcow2_l2_allocate_done(bs, l1_index, 0);

    return 0;



fail:

    trace_qcow2_l2_allocate_done(bs, l1_index, ret);

    qcow2_cache_put(bs, s->l2_table_cache, (void**) table);

    s->l1_table[l1_index] = old_l2_offset;

    return ret;

}
