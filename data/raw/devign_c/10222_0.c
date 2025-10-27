static int discard_single_l2(BlockDriverState *bs, uint64_t offset,

                             uint64_t nb_clusters, enum qcow2_discard_type type,

                             bool full_discard)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t *l2_table;

    int l2_index;

    int ret;

    int i;



    ret = get_cluster_table(bs, offset, &l2_table, &l2_index);

    if (ret < 0) {

        return ret;

    }



    /* Limit nb_clusters to one L2 table */

    nb_clusters = MIN(nb_clusters, s->l2_size - l2_index);

    assert(nb_clusters <= INT_MAX);



    for (i = 0; i < nb_clusters; i++) {

        uint64_t old_l2_entry;



        old_l2_entry = be64_to_cpu(l2_table[l2_index + i]);



        /*

         * If full_discard is false, make sure that a discarded area reads back

         * as zeroes for v3 images (we cannot do it for v2 without actually

         * writing a zero-filled buffer). We can skip the operation if the

         * cluster is already marked as zero, or if it's unallocated and we

         * don't have a backing file.

         *

         * TODO We might want to use bdrv_get_block_status(bs) here, but we're

         * holding s->lock, so that doesn't work today.

         *

         * If full_discard is true, the sector should not read back as zeroes,

         * but rather fall through to the backing file.

         */

        switch (qcow2_get_cluster_type(old_l2_entry)) {

        case QCOW2_CLUSTER_UNALLOCATED:

            if (full_discard || !bs->backing) {

                continue;

            }

            break;



        case QCOW2_CLUSTER_ZERO_PLAIN:

            if (!full_discard) {

                continue;

            }

            break;



        case QCOW2_CLUSTER_ZERO_ALLOC:

        case QCOW2_CLUSTER_NORMAL:

        case QCOW2_CLUSTER_COMPRESSED:

            break;



        default:

            abort();

        }



        /* First remove L2 entries */

        qcow2_cache_entry_mark_dirty(bs, s->l2_table_cache, l2_table);

        if (!full_discard && s->qcow_version >= 3) {

            l2_table[l2_index + i] = cpu_to_be64(QCOW_OFLAG_ZERO);

        } else {

            l2_table[l2_index + i] = cpu_to_be64(0);

        }



        /* Then decrease the refcount */

        qcow2_free_any_clusters(bs, old_l2_entry, 1, type);

    }



    qcow2_cache_put(bs, s->l2_table_cache, (void **) &l2_table);



    return nb_clusters;

}
