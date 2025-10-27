int qcow2_alloc_cluster_offset(BlockDriverState *bs, uint64_t offset,

    int n_start, int n_end, int *num, QCowL2Meta *m)

{

    BDRVQcowState *s = bs->opaque;

    int l2_index, ret;

    uint64_t l2_offset, *l2_table;

    int64_t cluster_offset;

    unsigned int nb_clusters, i = 0;

    QCowL2Meta *old_alloc;



    ret = get_cluster_table(bs, offset, &l2_table, &l2_offset, &l2_index);

    if (ret < 0) {

        return ret;

    }



again:

    nb_clusters = size_to_clusters(s, n_end << 9);



    nb_clusters = MIN(nb_clusters, s->l2_size - l2_index);



    cluster_offset = be64_to_cpu(l2_table[l2_index]);



    /* We keep all QCOW_OFLAG_COPIED clusters */



    if (cluster_offset & QCOW_OFLAG_COPIED) {

        nb_clusters = count_contiguous_clusters(nb_clusters, s->cluster_size,

                &l2_table[l2_index], 0, 0);



        cluster_offset &= ~QCOW_OFLAG_COPIED;

        m->nb_clusters = 0;



        goto out;

    }



    /* for the moment, multiple compressed clusters are not managed */



    if (cluster_offset & QCOW_OFLAG_COMPRESSED)

        nb_clusters = 1;



    /* how many available clusters ? */



    while (i < nb_clusters) {

        i += count_contiguous_clusters(nb_clusters - i, s->cluster_size,

                &l2_table[l2_index], i, 0);

        if ((i >= nb_clusters) || be64_to_cpu(l2_table[l2_index + i])) {

            break;

        }



        i += count_contiguous_free_clusters(nb_clusters - i,

                &l2_table[l2_index + i]);

        if (i >= nb_clusters) {

            break;

        }



        cluster_offset = be64_to_cpu(l2_table[l2_index + i]);



        if ((cluster_offset & QCOW_OFLAG_COPIED) ||

                (cluster_offset & QCOW_OFLAG_COMPRESSED))

            break;

    }

    assert(i <= nb_clusters);

    nb_clusters = i;



    /*

     * Check if there already is an AIO write request in flight which allocates

     * the same cluster. In this case we need to wait until the previous

     * request has completed and updated the L2 table accordingly.

     */

    QLIST_FOREACH(old_alloc, &s->cluster_allocs, next_in_flight) {



        uint64_t end_offset = offset + nb_clusters * s->cluster_size;

        uint64_t old_offset = old_alloc->offset;

        uint64_t old_end_offset = old_alloc->offset +

            old_alloc->nb_clusters * s->cluster_size;



        if (end_offset < old_offset || offset > old_end_offset) {

            /* No intersection */

        } else {

            if (offset < old_offset) {

                /* Stop at the start of a running allocation */

                nb_clusters = (old_offset - offset) >> s->cluster_bits;

            } else {

                nb_clusters = 0;

            }



            if (nb_clusters == 0) {

                /* Wait for the dependency to complete. We need to recheck

                 * the free/allocated clusters when we continue. */

                qemu_co_mutex_unlock(&s->lock);

                qemu_co_queue_wait(&old_alloc->dependent_requests);

                qemu_co_mutex_lock(&s->lock);

                goto again;

            }

        }

    }



    if (!nb_clusters) {

        abort();

    }



    QLIST_INSERT_HEAD(&s->cluster_allocs, m, next_in_flight);



    /* allocate a new cluster */



    cluster_offset = qcow2_alloc_clusters(bs, nb_clusters * s->cluster_size);

    if (cluster_offset < 0) {

        ret = cluster_offset;

        goto fail;

    }



    /* save info needed for meta data update */

    m->offset = offset;

    m->n_start = n_start;

    m->nb_clusters = nb_clusters;



out:

    ret = qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

    if (ret < 0) {

        goto fail_put;

    }



    m->nb_available = MIN(nb_clusters << (s->cluster_bits - 9), n_end);

    m->cluster_offset = cluster_offset;



    *num = m->nb_available - n_start;



    return 0;



fail:

    qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

fail_put:

    QLIST_REMOVE(m, next_in_flight);

    return ret;

}
