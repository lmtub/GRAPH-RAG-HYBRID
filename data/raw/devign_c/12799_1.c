int qcow2_expand_zero_clusters(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l1_table = NULL;

    uint64_t nb_clusters;

    uint8_t *expanded_clusters;

    int ret;

    int i, j;



    nb_clusters = size_to_clusters(s, bs->file->total_sectors *

                                   BDRV_SECTOR_SIZE);

    expanded_clusters = g_malloc0((nb_clusters + 7) / 8);



    ret = expand_zero_clusters_in_l1(bs, s->l1_table, s->l1_size,

                                     &expanded_clusters, &nb_clusters);

    if (ret < 0) {

        goto fail;

    }



    /* Inactive L1 tables may point to active L2 tables - therefore it is

     * necessary to flush the L2 table cache before trying to access the L2

     * tables pointed to by inactive L1 entries (else we might try to expand

     * zero clusters that have already been expanded); furthermore, it is also

     * necessary to empty the L2 table cache, since it may contain tables which

     * are now going to be modified directly on disk, bypassing the cache.

     * qcow2_cache_empty() does both for us. */

    ret = qcow2_cache_empty(bs, s->l2_table_cache);

    if (ret < 0) {

        goto fail;

    }



    for (i = 0; i < s->nb_snapshots; i++) {

        int l1_sectors = (s->snapshots[i].l1_size * sizeof(uint64_t) +

                BDRV_SECTOR_SIZE - 1) / BDRV_SECTOR_SIZE;



        l1_table = g_realloc(l1_table, l1_sectors * BDRV_SECTOR_SIZE);



        ret = bdrv_read(bs->file, s->snapshots[i].l1_table_offset /

                BDRV_SECTOR_SIZE, (void *)l1_table, l1_sectors);

        if (ret < 0) {

            goto fail;

        }



        for (j = 0; j < s->snapshots[i].l1_size; j++) {

            be64_to_cpus(&l1_table[j]);

        }



        ret = expand_zero_clusters_in_l1(bs, l1_table, s->snapshots[i].l1_size,

                                         &expanded_clusters, &nb_clusters);

        if (ret < 0) {

            goto fail;

        }

    }



    ret = 0;



fail:

    g_free(expanded_clusters);

    g_free(l1_table);

    return ret;

}
