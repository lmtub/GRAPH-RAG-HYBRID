void qcow2_free_any_clusters(BlockDriverState *bs, uint64_t l2_entry,

                             int nb_clusters, enum qcow2_discard_type type)

{

    BDRVQcow2State *s = bs->opaque;



    switch (qcow2_get_cluster_type(l2_entry)) {

    case QCOW2_CLUSTER_COMPRESSED:

        {

            int nb_csectors;

            nb_csectors = ((l2_entry >> s->csize_shift) &

                           s->csize_mask) + 1;

            qcow2_free_clusters(bs,

                (l2_entry & s->cluster_offset_mask) & ~511,

                nb_csectors * 512, type);

        }

        break;

    case QCOW2_CLUSTER_NORMAL:

    case QCOW2_CLUSTER_ZERO:

        if (l2_entry & L2E_OFFSET_MASK) {

            if (offset_into_cluster(s, l2_entry & L2E_OFFSET_MASK)) {

                qcow2_signal_corruption(bs, false, -1, -1,

                                        "Cannot free unaligned cluster %#llx",

                                        l2_entry & L2E_OFFSET_MASK);

            } else {

                qcow2_free_clusters(bs, l2_entry & L2E_OFFSET_MASK,

                                    nb_clusters << s->cluster_bits, type);

            }

        }

        break;

    case QCOW2_CLUSTER_UNALLOCATED:

        break;

    default:

        abort();

    }

}
