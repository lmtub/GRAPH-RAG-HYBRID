static int get_cluster_offset(BlockDriverState *bs,

                              uint64_t offset, int allocate,

                              int compressed_size,

                              int n_start, int n_end, uint64_t *result)

{

    BDRVQcowState *s = bs->opaque;

    int min_index, i, j, l1_index, l2_index, ret;

    uint64_t l2_offset, *l2_table, cluster_offset, tmp;

    uint32_t min_count;

    int new_l2_table;



    *result = 0;

    l1_index = offset >> (s->l2_bits + s->cluster_bits);

    l2_offset = s->l1_table[l1_index];

    new_l2_table = 0;

    if (!l2_offset) {

        if (!allocate)

            return 0;

        /* allocate a new l2 entry */

        l2_offset = bdrv_getlength(bs->file->bs);

        /* round to cluster size */

        l2_offset = (l2_offset + s->cluster_size - 1) & ~(s->cluster_size - 1);

        /* update the L1 entry */

        s->l1_table[l1_index] = l2_offset;

        tmp = cpu_to_be64(l2_offset);

        ret = bdrv_pwrite_sync(bs->file,

                               s->l1_table_offset + l1_index * sizeof(tmp),

                               &tmp, sizeof(tmp));

        if (ret < 0) {

            return ret;

        }

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

        ret = bdrv_pwrite_sync(bs->file, l2_offset, l2_table,

                               s->l2_size * sizeof(uint64_t));

        if (ret < 0) {

            return ret;

        }

    } else {

        ret = bdrv_pread(bs->file, l2_offset, l2_table,

                         s->l2_size * sizeof(uint64_t));

        if (ret < 0) {

            return ret;

        }

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

            if (decompress_cluster(bs, cluster_offset) < 0) {

                return -EIO;

            }

            cluster_offset = bdrv_getlength(bs->file->bs);

            cluster_offset = (cluster_offset + s->cluster_size - 1) &

                ~(s->cluster_size - 1);

            /* write the cluster content */

            ret = bdrv_pwrite(bs->file, cluster_offset, s->cluster_cache,

                              s->cluster_size);

            if (ret < 0) {

                return ret;

            }

        } else {

            cluster_offset = bdrv_getlength(bs->file->bs);

            if (allocate == 1) {

                /* round to cluster size */

                cluster_offset = (cluster_offset + s->cluster_size - 1) &

                    ~(s->cluster_size - 1);

                bdrv_truncate(bs->file, cluster_offset + s->cluster_size,

                              PREALLOC_MODE_OFF, NULL);

                /* if encrypted, we must initialize the cluster

                   content which won't be written */

                if (bs->encrypted &&

                    (n_end - n_start) < s->cluster_sectors) {

                    uint64_t start_sect;

                    assert(s->crypto);

                    start_sect = (offset & ~(s->cluster_size - 1)) >> 9;

                    for(i = 0; i < s->cluster_sectors; i++) {

                        if (i < n_start || i >= n_end) {

                            memset(s->cluster_data, 0x00, 512);

                            if (qcrypto_block_encrypt(s->crypto, start_sect + i,

                                                      s->cluster_data,

                                                      BDRV_SECTOR_SIZE,

                                                      NULL) < 0) {

                                return -EIO;

                            }

                            ret = bdrv_pwrite(bs->file,

                                              cluster_offset + i * 512,

                                              s->cluster_data, 512);

                            if (ret < 0) {

                                return ret;

                            }

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

        ret = bdrv_pwrite_sync(bs->file, l2_offset + l2_index * sizeof(tmp),

                               &tmp, sizeof(tmp));

        if (ret < 0) {

            return ret;

        }

    }

    *result = cluster_offset;

    return 1;

}
