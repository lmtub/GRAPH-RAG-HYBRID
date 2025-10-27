static uint64_t get_cluster_offset(BlockDriverState *bs,

                                   uint64_t offset, int allocate,

                                   int compressed_size,

                                   int n_start, int n_end)

{

    BDRVQcowState *s = bs->opaque;

    int min_index, i, j, l1_index, l2_index;

    uint64_t l2_offset, *l2_table, cluster_offset, tmp;

    uint32_t min_count;

    int new_l2_table;



    l1_index = offset >> (s->l2_bits + s->cluster_bits);

    l2_offset = s->l1_table[l1_index];

    new_l2_table = 0;

    if (!l2_offset) {

        if (!allocate)

            return 0;

        /* allocate a new l2 entry */

        l2_offset = bdrv_getlength(bs->file);

        /* round to cluster size */

        l2_offset = (l2_offset + s->cluster_size - 1) & ~(s->cluster_size - 1);

        /* update the L1 entry */

        s->l1_table[l1_index] = l2_offset;

        tmp = cpu_to_be64(l2_offset);

        if (bdrv_pwrite(bs->file, s->l1_table_offset + l1_index * sizeof(tmp),

                        &tmp, sizeof(tmp)) != sizeof(tmp))

            return 0;

        new_l2_table = 1;

    }

    for(i = 0; i < L2_CACHE_SIZE; i++) {

        if (l2_offset == s->l2_cache_offsets[i]) {

            /* increment the hit count */

            if (++s->l2_cache_counts[i] == 0xffffffff) {

                for(j = 0; j < L2_CACHE_SIZE; j++) {

                    s->l2_cache_counts[j] >>= 1;

                }

            }

            l2_table = s->l2_cache + (i << s->l2_bits);

            goto found;

        }

    }

    /* not found: load a new entry in the least used one */

    min_index = 0;

    min_count = 0xffffffff;

    for(i = 0; i < L2_CACHE_SIZE; i++) {

        if (s->l2_cache_counts[i] < min_count) {

            min_count = s->l2_cache_counts[i];

            min_index = i;

        }

    }

    l2_table = s->l2_cache + (min_index << s->l2_bits);

    if (new_l2_table) {

        memset(l2_table, 0, s->l2_size * sizeof(uint64_t));

        if (bdrv_pwrite(bs->file, l2_offset, l2_table, s->l2_size * sizeof(uint64_t)) !=

            s->l2_size * sizeof(uint64_t))

            return 0;

    } else {

        if (bdrv_pread(bs->file, l2_offset, l2_table, s->l2_size * sizeof(uint64_t)) !=

            s->l2_size * sizeof(uint64_t))

            return 0;

    }

    s->l2_cache_offsets[min_index] = l2_offset;

    s->l2_cache_counts[min_index] = 1;

 found:

    l2_index = (offset >> s->cluster_bits) & (s->l2_size - 1);

    cluster_offset = be64_to_cpu(l2_table[l2_index]);

    if (!cluster_offset ||

        ((cluster_offset & QCOW_OFLAG_COMPRESSED) && allocate == 1)) {

        if (!allocate)

            return 0;

        /* allocate a new cluster */

        if ((cluster_offset & QCOW_OFLAG_COMPRESSED) &&

            (n_end - n_start) < s->cluster_sectors) {

            /* if the cluster is already compressed, we must

               decompress it in the case it is not completely

               overwritten */

            if (decompress_cluster(bs, cluster_offset) < 0)

                return 0;

            cluster_offset = bdrv_getlength(bs->file);

            cluster_offset = (cluster_offset + s->cluster_size - 1) &

                ~(s->cluster_size - 1);

            /* write the cluster content */

            if (bdrv_pwrite(bs->file, cluster_offset, s->cluster_cache, s->cluster_size) !=

                s->cluster_size)

                return -1;

        } else {

            cluster_offset = bdrv_getlength(bs->file);

            if (allocate == 1) {

                /* round to cluster size */

                cluster_offset = (cluster_offset + s->cluster_size - 1) &

                    ~(s->cluster_size - 1);

                bdrv_truncate(bs->file, cluster_offset + s->cluster_size);

                /* if encrypted, we must initialize the cluster

                   content which won't be written */

                if (s->crypt_method &&

                    (n_end - n_start) < s->cluster_sectors) {

                    uint64_t start_sect;

                    start_sect = (offset & ~(s->cluster_size - 1)) >> 9;

                    memset(s->cluster_data + 512, 0x00, 512);

                    for(i = 0; i < s->cluster_sectors; i++) {

                        if (i < n_start || i >= n_end) {

                            encrypt_sectors(s, start_sect + i,

                                            s->cluster_data,

                                            s->cluster_data + 512, 1, 1,

                                            &s->aes_encrypt_key);

                            if (bdrv_pwrite(bs->file, cluster_offset + i * 512,

                                            s->cluster_data, 512) != 512)

                                return -1;

                        }

                    }

                }

            } else if (allocate == 2) {

                cluster_offset |= QCOW_OFLAG_COMPRESSED |

                    (uint64_t)compressed_size << (63 - s->cluster_bits);

            }

        }

        /* update L2 table */

        tmp = cpu_to_be64(cluster_offset);

        l2_table[l2_index] = tmp;

        if (bdrv_pwrite(bs->file,

                        l2_offset + l2_index * sizeof(tmp), &tmp, sizeof(tmp)) != sizeof(tmp))

            return 0;

    }

    return cluster_offset;

}
