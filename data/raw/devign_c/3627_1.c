static uint64_t alloc_cluster_offset(BlockDriverState *bs,

                                     uint64_t offset,

                                     int n_start, int n_end,

                                     int *num, QCowL2Meta *m)

{

    BDRVQcowState *s = bs->opaque;

    int l2_index, ret;

    uint64_t l2_offset, *l2_table, cluster_offset;

    int nb_clusters, i = 0;



    ret = get_cluster_table(bs, offset, &l2_table, &l2_offset, &l2_index);

    if (ret == 0)

        return 0;



    nb_clusters = size_to_clusters(s, n_end << 9);



    nb_clusters = MIN(nb_clusters, s->l2_size - l2_index);



    cluster_offset = be64_to_cpu(l2_table[l2_index]);



    /* We keep all QCOW_OFLAG_COPIED clusters */



    if (cluster_offset & QCOW_OFLAG_COPIED) {

        nb_clusters = count_contiguous_clusters(nb_clusters, s->cluster_size,

                &l2_table[l2_index], 0);



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

                &l2_table[l2_index + i], 0);



        if(be64_to_cpu(l2_table[l2_index + i]))

            break;



        i += count_contiguous_free_clusters(nb_clusters - i,

                &l2_table[l2_index + i]);



        cluster_offset = be64_to_cpu(l2_table[l2_index + i]);



        if ((cluster_offset & QCOW_OFLAG_COPIED) ||

                (cluster_offset & QCOW_OFLAG_COMPRESSED))

            break;

    }

    nb_clusters = i;



    /* allocate a new cluster */



    cluster_offset = alloc_clusters(bs, nb_clusters * s->cluster_size);



    /* save info needed for meta data update */

    m->offset = offset;

    m->n_start = n_start;

    m->nb_clusters = nb_clusters;



out:

    m->nb_available = MIN(nb_clusters << (s->cluster_bits - 9), n_end);



    *num = m->nb_available - n_start;



    return cluster_offset;

}
