static int rebuild_refcount_structure(BlockDriverState *bs,

                                      BdrvCheckResult *res,

                                      void **refcount_table,

                                      int64_t *nb_clusters)

{

    BDRVQcow2State *s = bs->opaque;

    int64_t first_free_cluster = 0, reftable_offset = -1, cluster = 0;

    int64_t refblock_offset, refblock_start, refblock_index;

    uint32_t reftable_size = 0;

    uint64_t *on_disk_reftable = NULL;

    void *on_disk_refblock;

    int ret = 0;

    struct {

        uint64_t reftable_offset;

        uint32_t reftable_clusters;

    } QEMU_PACKED reftable_offset_and_clusters;



    qcow2_cache_empty(bs, s->refcount_block_cache);



write_refblocks:

    for (; cluster < *nb_clusters; cluster++) {

        if (!s->get_refcount(*refcount_table, cluster)) {

            continue;

        }



        refblock_index = cluster >> s->refcount_block_bits;

        refblock_start = refblock_index << s->refcount_block_bits;



        /* Don't allocate a cluster in a refblock already written to disk */

        if (first_free_cluster < refblock_start) {

            first_free_cluster = refblock_start;

        }

        refblock_offset = alloc_clusters_imrt(bs, 1, refcount_table,

                                              nb_clusters, &first_free_cluster);

        if (refblock_offset < 0) {

            fprintf(stderr, "ERROR allocating refblock: %s\n",

                    strerror(-refblock_offset));

            res->check_errors++;

            ret = refblock_offset;

            goto fail;

        }



        if (reftable_size <= refblock_index) {

            uint32_t old_reftable_size = reftable_size;

            uint64_t *new_on_disk_reftable;



            reftable_size = ROUND_UP((refblock_index + 1) * sizeof(uint64_t),

                                     s->cluster_size) / sizeof(uint64_t);

            new_on_disk_reftable = g_try_realloc(on_disk_reftable,

                                                 reftable_size *

                                                 sizeof(uint64_t));

            if (!new_on_disk_reftable) {

                res->check_errors++;

                ret = -ENOMEM;

                goto fail;

            }

            on_disk_reftable = new_on_disk_reftable;



            memset(on_disk_reftable + old_reftable_size, 0,

                   (reftable_size - old_reftable_size) * sizeof(uint64_t));



            /* The offset we have for the reftable is now no longer valid;

             * this will leak that range, but we can easily fix that by running

             * a leak-fixing check after this rebuild operation */

            reftable_offset = -1;

        }

        on_disk_reftable[refblock_index] = refblock_offset;



        /* If this is apparently the last refblock (for now), try to squeeze the

         * reftable in */

        if (refblock_index == (*nb_clusters - 1) >> s->refcount_block_bits &&

            reftable_offset < 0)

        {

            uint64_t reftable_clusters = size_to_clusters(s, reftable_size *

                                                          sizeof(uint64_t));

            reftable_offset = alloc_clusters_imrt(bs, reftable_clusters,

                                                  refcount_table, nb_clusters,

                                                  &first_free_cluster);

            if (reftable_offset < 0) {

                fprintf(stderr, "ERROR allocating reftable: %s\n",

                        strerror(-reftable_offset));

                res->check_errors++;

                ret = reftable_offset;

                goto fail;

            }

        }



        ret = qcow2_pre_write_overlap_check(bs, 0, refblock_offset,

                                            s->cluster_size);

        if (ret < 0) {

            fprintf(stderr, "ERROR writing refblock: %s\n", strerror(-ret));

            goto fail;

        }



        /* The size of *refcount_table is always cluster-aligned, therefore the

         * write operation will not overflow */

        on_disk_refblock = (void *)((char *) *refcount_table +

                                    refblock_index * s->cluster_size);



        ret = bdrv_write(bs->file, refblock_offset / BDRV_SECTOR_SIZE,

                         on_disk_refblock, s->cluster_sectors);

        if (ret < 0) {

            fprintf(stderr, "ERROR writing refblock: %s\n", strerror(-ret));

            goto fail;

        }



        /* Go to the end of this refblock */

        cluster = refblock_start + s->refcount_block_size - 1;

    }



    if (reftable_offset < 0) {

        uint64_t post_refblock_start, reftable_clusters;



        post_refblock_start = ROUND_UP(*nb_clusters, s->refcount_block_size);

        reftable_clusters = size_to_clusters(s,

                                             reftable_size * sizeof(uint64_t));

        /* Not pretty but simple */

        if (first_free_cluster < post_refblock_start) {

            first_free_cluster = post_refblock_start;

        }

        reftable_offset = alloc_clusters_imrt(bs, reftable_clusters,

                                              refcount_table, nb_clusters,

                                              &first_free_cluster);

        if (reftable_offset < 0) {

            fprintf(stderr, "ERROR allocating reftable: %s\n",

                    strerror(-reftable_offset));

            res->check_errors++;

            ret = reftable_offset;

            goto fail;

        }



        goto write_refblocks;

    }



    assert(on_disk_reftable);



    for (refblock_index = 0; refblock_index < reftable_size; refblock_index++) {

        cpu_to_be64s(&on_disk_reftable[refblock_index]);

    }



    ret = qcow2_pre_write_overlap_check(bs, 0, reftable_offset,

                                        reftable_size * sizeof(uint64_t));

    if (ret < 0) {

        fprintf(stderr, "ERROR writing reftable: %s\n", strerror(-ret));

        goto fail;

    }



    assert(reftable_size < INT_MAX / sizeof(uint64_t));

    ret = bdrv_pwrite(bs->file, reftable_offset, on_disk_reftable,

                      reftable_size * sizeof(uint64_t));

    if (ret < 0) {

        fprintf(stderr, "ERROR writing reftable: %s\n", strerror(-ret));

        goto fail;

    }



    /* Enter new reftable into the image header */

    reftable_offset_and_clusters.reftable_offset = cpu_to_be64(reftable_offset);

    reftable_offset_and_clusters.reftable_clusters =

        cpu_to_be32(size_to_clusters(s, reftable_size * sizeof(uint64_t)));

    ret = bdrv_pwrite_sync(bs->file,

                           offsetof(QCowHeader, refcount_table_offset),

                           &reftable_offset_and_clusters,

                           sizeof(reftable_offset_and_clusters));

    if (ret < 0) {

        fprintf(stderr, "ERROR setting reftable: %s\n", strerror(-ret));

        goto fail;

    }



    for (refblock_index = 0; refblock_index < reftable_size; refblock_index++) {

        be64_to_cpus(&on_disk_reftable[refblock_index]);

    }

    s->refcount_table = on_disk_reftable;

    s->refcount_table_offset = reftable_offset;

    s->refcount_table_size = reftable_size;

    update_max_refcount_table_index(s);



    return 0;



fail:

    g_free(on_disk_reftable);

    return ret;

}
