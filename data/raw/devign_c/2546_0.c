static int do_alloc_cluster_offset(BlockDriverState *bs, uint64_t guest_offset,

    uint64_t *host_offset, unsigned int *nb_clusters)

{

    BDRVQcowState *s = bs->opaque;

    int ret;



    trace_qcow2_do_alloc_clusters_offset(qemu_coroutine_self(), guest_offset,

                                         *host_offset, *nb_clusters);



    ret = handle_dependencies(bs, guest_offset, nb_clusters);

    if (ret < 0) {

        return ret;

    }



    /* Allocate new clusters */

    trace_qcow2_cluster_alloc_phys(qemu_coroutine_self());

    if (*host_offset == 0) {

        int64_t cluster_offset =

            qcow2_alloc_clusters(bs, *nb_clusters * s->cluster_size);

        if (cluster_offset < 0) {

            return cluster_offset;

        }

        *host_offset = cluster_offset;

        return 0;

    } else {

        ret = qcow2_alloc_clusters_at(bs, *host_offset, *nb_clusters);

        if (ret < 0) {

            return ret;

        }

        *nb_clusters = ret;

        return 0;

    }

}
