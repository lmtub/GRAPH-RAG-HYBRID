static int count_contiguous_free_clusters(int nb_clusters, uint64_t *l2_table)

{

    int i;



    for (i = 0; i < nb_clusters; i++) {

        int type = qcow2_get_cluster_type(be64_to_cpu(l2_table[i]));



        if (type != QCOW2_CLUSTER_UNALLOCATED) {

            break;

        }

    }



    return i;

}
