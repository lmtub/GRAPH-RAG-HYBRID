static int count_contiguous_clusters(int nb_clusters, int cluster_size,

        uint64_t *l2_table, uint64_t stop_flags)

{

    int i;

    uint64_t mask = stop_flags | L2E_OFFSET_MASK | QCOW_OFLAG_COMPRESSED;

    uint64_t first_entry = be64_to_cpu(l2_table[0]);

    uint64_t offset = first_entry & mask;



    if (!offset)

        return 0;



    assert(qcow2_get_cluster_type(first_entry) != QCOW2_CLUSTER_COMPRESSED);



    for (i = 0; i < nb_clusters; i++) {

        uint64_t l2_entry = be64_to_cpu(l2_table[i]) & mask;

        if (offset + (uint64_t) i * cluster_size != l2_entry) {

            break;

        }

    }



	return i;

}
