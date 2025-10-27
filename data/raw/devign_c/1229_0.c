static uint64_t get_cluster_offset(BlockDriverState *bs,

                                    VmdkExtent *extent,

                                    VmdkMetaData *m_data,

                                    uint64_t offset, int allocate)

{

    unsigned int l1_index, l2_offset, l2_index;

    int min_index, i, j;

    uint32_t min_count, *l2_table, tmp = 0;

    uint64_t cluster_offset;



    if (m_data)

        m_data->valid = 0;



    l1_index = (offset >> 9) / extent->l1_entry_sectors;

    if (l1_index >= extent->l1_size) {

        return 0;

    }

    l2_offset = extent->l1_table[l1_index];

    if (!l2_offset) {

        return 0;

    }

    for (i = 0; i < L2_CACHE_SIZE; i++) {

        if (l2_offset == extent->l2_cache_offsets[i]) {

            /* increment the hit count */

            if (++extent->l2_cache_counts[i] == 0xffffffff) {

                for (j = 0; j < L2_CACHE_SIZE; j++) {

                    extent->l2_cache_counts[j] >>= 1;

                }

            }

            l2_table = extent->l2_cache + (i * extent->l2_size);

            goto found;

        }

    }

    /* not found: load a new entry in the least used one */

    min_index = 0;

    min_count = 0xffffffff;

    for (i = 0; i < L2_CACHE_SIZE; i++) {

        if (extent->l2_cache_counts[i] < min_count) {

            min_count = extent->l2_cache_counts[i];

            min_index = i;

        }

    }

    l2_table = extent->l2_cache + (min_index * extent->l2_size);

    if (bdrv_pread(

                extent->file,

                (int64_t)l2_offset * 512,

                l2_table,

                extent->l2_size * sizeof(uint32_t)

            ) != extent->l2_size * sizeof(uint32_t)) {

        return 0;

    }



    extent->l2_cache_offsets[min_index] = l2_offset;

    extent->l2_cache_counts[min_index] = 1;

 found:

    l2_index = ((offset >> 9) / extent->cluster_sectors) % extent->l2_size;

    cluster_offset = le32_to_cpu(l2_table[l2_index]);



    if (!cluster_offset) {

        if (!allocate)

            return 0;



        // Avoid the L2 tables update for the images that have snapshots.

        cluster_offset = bdrv_getlength(extent->file);

        bdrv_truncate(

            extent->file,

            cluster_offset + (extent->cluster_sectors << 9)

        );



        cluster_offset >>= 9;

        tmp = cpu_to_le32(cluster_offset);

        l2_table[l2_index] = tmp;



        /* First of all we write grain itself, to avoid race condition

         * that may to corrupt the image.

         * This problem may occur because of insufficient space on host disk

         * or inappropriate VM shutdown.

         */

        if (get_whole_cluster(

                bs, extent, cluster_offset, offset, allocate) == -1)

            return 0;



        if (m_data) {

            m_data->offset = tmp;

            m_data->l1_index = l1_index;

            m_data->l2_index = l2_index;

            m_data->l2_offset = l2_offset;

            m_data->valid = 1;

        }

    }

    cluster_offset <<= 9;

    return cluster_offset;

}
