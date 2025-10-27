uint64_t qcow2_get_cluster_offset(BlockDriverState *bs, uint64_t offset,

    int *num)

{

    BDRVQcowState *s = bs->opaque;

    int l1_index, l2_index;

    uint64_t l2_offset, *l2_table, cluster_offset;

    int l1_bits, c;

    int index_in_cluster, nb_available, nb_needed, nb_clusters;



    index_in_cluster = (offset >> 9) & (s->cluster_sectors - 1);

    nb_needed = *num + index_in_cluster;



    l1_bits = s->l2_bits + s->cluster_bits;



    /* compute how many bytes there are between the offset and

     * the end of the l1 entry

     */



    nb_available = (1 << l1_bits) - (offset & ((1 << l1_bits) - 1));



    /* compute the number of available sectors */



    nb_available = (nb_available >> 9) + index_in_cluster;



    if (nb_needed > nb_available) {

        nb_needed = nb_available;

    }



    cluster_offset = 0;



    /* seek the the l2 offset in the l1 table */



    l1_index = offset >> l1_bits;

    if (l1_index >= s->l1_size)

        goto out;



    l2_offset = s->l1_table[l1_index];



    /* seek the l2 table of the given l2 offset */



    if (!l2_offset)

        goto out;



    /* load the l2 table in memory */



    l2_offset &= ~QCOW_OFLAG_COPIED;

    l2_table = l2_load(bs, l2_offset);

    if (l2_table == NULL)

        return 0;



    /* find the cluster offset for the given disk offset */



    l2_index = (offset >> s->cluster_bits) & (s->l2_size - 1);

    cluster_offset = be64_to_cpu(l2_table[l2_index]);

    nb_clusters = size_to_clusters(s, nb_needed << 9);



    if (!cluster_offset) {

        /* how many empty clusters ? */

        c = count_contiguous_free_clusters(nb_clusters, &l2_table[l2_index]);

    } else {

        /* how many allocated clusters ? */

        c = count_contiguous_clusters(nb_clusters, s->cluster_size,

                &l2_table[l2_index], 0, QCOW_OFLAG_COPIED);

    }



   nb_available = (c * s->cluster_sectors);

out:

    if (nb_available > nb_needed)

        nb_available = nb_needed;



    *num = nb_available - index_in_cluster;



    return cluster_offset & ~QCOW_OFLAG_COPIED;

}
