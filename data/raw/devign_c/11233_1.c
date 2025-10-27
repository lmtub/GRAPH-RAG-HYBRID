int qcow2_alloc_cluster_link_l2(BlockDriverState *bs, QCowL2Meta *m)

{

    BDRVQcowState *s = bs->opaque;

    int i, j = 0, l2_index, ret;

    uint64_t *old_cluster, *l2_table;

    uint64_t cluster_offset = m->alloc_offset;



    trace_qcow2_cluster_link_l2(qemu_coroutine_self(), m->nb_clusters);

    assert(m->nb_clusters > 0);



    old_cluster = g_malloc(m->nb_clusters * sizeof(uint64_t));



    /* copy content of unmodified sectors */

    ret = perform_cow(bs, m, &m->cow_start);

    if (ret < 0) {

        goto err;

    }



    ret = perform_cow(bs, m, &m->cow_end);

    if (ret < 0) {

        goto err;

    }



    /* Update L2 table. */

    if (s->use_lazy_refcounts) {

        qcow2_mark_dirty(bs);

    }

    if (qcow2_need_accurate_refcounts(s)) {

        qcow2_cache_set_dependency(bs, s->l2_table_cache,

                                   s->refcount_block_cache);

    }



    ret = get_cluster_table(bs, m->offset, &l2_table, &l2_index);

    if (ret < 0) {

        goto err;

    }

    qcow2_cache_entry_mark_dirty(s->l2_table_cache, l2_table);




    for (i = 0; i < m->nb_clusters; i++) {

        /* if two concurrent writes happen to the same unallocated cluster

	 * each write allocates separate cluster and writes data concurrently.

	 * The first one to complete updates l2 table with pointer to its

	 * cluster the second one has to do RMW (which is done above by

	 * copy_sectors()), update l2 table with its cluster pointer and free

	 * old cluster. This is what this loop does */

        if(l2_table[l2_index + i] != 0)

            old_cluster[j++] = l2_table[l2_index + i];



        l2_table[l2_index + i] = cpu_to_be64((cluster_offset +

                    (i << s->cluster_bits)) | QCOW_OFLAG_COPIED);

     }





    ret = qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

    if (ret < 0) {

        goto err;

    }



    /*

     * If this was a COW, we need to decrease the refcount of the old cluster.

     * Also flush bs->file to get the right order for L2 and refcount update.

     *

     * Don't discard clusters that reach a refcount of 0 (e.g. compressed

     * clusters), the next write will reuse them anyway.

     */

    if (j != 0) {

        for (i = 0; i < j; i++) {

            qcow2_free_any_clusters(bs, be64_to_cpu(old_cluster[i]), 1,

                                    QCOW2_DISCARD_NEVER);

        }

    }



    ret = 0;

err:

    g_free(old_cluster);

    return ret;

 }