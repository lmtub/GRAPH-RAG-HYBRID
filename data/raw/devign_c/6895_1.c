static int alloc_refcount_block(BlockDriverState *bs,

                                int64_t cluster_index, void **refcount_block)

{

    BDRVQcow2State *s = bs->opaque;

    unsigned int refcount_table_index;

    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC);



    /* Find the refcount block for the given cluster */

    refcount_table_index = cluster_index >> s->refcount_block_bits;



    if (refcount_table_index < s->refcount_table_size) {



        uint64_t refcount_block_offset =

            s->refcount_table[refcount_table_index] & REFT_OFFSET_MASK;



        /* If it's already there, we're done */

        if (refcount_block_offset) {

            if (offset_into_cluster(s, refcount_block_offset)) {

                qcow2_signal_corruption(bs, true, -1, -1, "Refblock offset %#"

                                        PRIx64 " unaligned (reftable index: "

                                        "%#x)", refcount_block_offset,

                                        refcount_table_index);

                return -EIO;

            }



             return load_refcount_block(bs, refcount_block_offset,

                                        refcount_block);

        }

    }



    /*

     * If we came here, we need to allocate something. Something is at least

     * a cluster for the new refcount block. It may also include a new refcount

     * table if the old refcount table is too small.

     *

     * Note that allocating clusters here needs some special care:

     *

     * - We can't use the normal qcow2_alloc_clusters(), it would try to

     *   increase the refcount and very likely we would end up with an endless

     *   recursion. Instead we must place the refcount blocks in a way that

     *   they can describe them themselves.

     *

     * - We need to consider that at this point we are inside update_refcounts

     *   and potentially doing an initial refcount increase. This means that

     *   some clusters have already been allocated by the caller, but their

     *   refcount isn't accurate yet. If we allocate clusters for metadata, we

     *   need to return -EAGAIN to signal the caller that it needs to restart

     *   the search for free clusters.

     *

     * - alloc_clusters_noref and qcow2_free_clusters may load a different

     *   refcount block into the cache

     */



    *refcount_block = NULL;



    /* We write to the refcount table, so we might depend on L2 tables */

    ret = qcow2_cache_flush(bs, s->l2_table_cache);

    if (ret < 0) {

        return ret;

    }



    /* Allocate the refcount block itself and mark it as used */

    int64_t new_block = alloc_clusters_noref(bs, s->cluster_size);

    if (new_block < 0) {

        return new_block;

    }



#ifdef DEBUG_ALLOC2

    fprintf(stderr, "qcow2: Allocate refcount block %d for %" PRIx64

        " at %" PRIx64 "\n",

        refcount_table_index, cluster_index << s->cluster_bits, new_block);

#endif



    if (in_same_refcount_block(s, new_block, cluster_index << s->cluster_bits)) {

        /* Zero the new refcount block before updating it */

        ret = qcow2_cache_get_empty(bs, s->refcount_block_cache, new_block,

                                    refcount_block);

        if (ret < 0) {

            goto fail_block;

        }



        memset(*refcount_block, 0, s->cluster_size);



        /* The block describes itself, need to update the cache */

        int block_index = (new_block >> s->cluster_bits) &

            (s->refcount_block_size - 1);

        s->set_refcount(*refcount_block, block_index, 1);

    } else {

        /* Described somewhere else. This can recurse at most twice before we

         * arrive at a block that describes itself. */

        ret = update_refcount(bs, new_block, s->cluster_size, 1, false,

                              QCOW2_DISCARD_NEVER);

        if (ret < 0) {

            goto fail_block;

        }



        ret = qcow2_cache_flush(bs, s->refcount_block_cache);

        if (ret < 0) {

            goto fail_block;

        }



        /* Initialize the new refcount block only after updating its refcount,

         * update_refcount uses the refcount cache itself */

        ret = qcow2_cache_get_empty(bs, s->refcount_block_cache, new_block,

                                    refcount_block);

        if (ret < 0) {

            goto fail_block;

        }



        memset(*refcount_block, 0, s->cluster_size);

    }



    /* Now the new refcount block needs to be written to disk */

    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_WRITE);

    qcow2_cache_entry_mark_dirty(bs, s->refcount_block_cache, *refcount_block);

    ret = qcow2_cache_flush(bs, s->refcount_block_cache);

    if (ret < 0) {

        goto fail_block;

    }



    /* If the refcount table is big enough, just hook the block up there */

    if (refcount_table_index < s->refcount_table_size) {

        uint64_t data64 = cpu_to_be64(new_block);

        BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_HOOKUP);

        ret = bdrv_pwrite_sync(bs->file->bs,

            s->refcount_table_offset + refcount_table_index * sizeof(uint64_t),

            &data64, sizeof(data64));

        if (ret < 0) {

            goto fail_block;

        }



        s->refcount_table[refcount_table_index] = new_block;



        /* The new refcount block may be where the caller intended to put its

         * data, so let it restart the search. */

        return -EAGAIN;

    }



    qcow2_cache_put(bs, s->refcount_block_cache, refcount_block);



    /*

     * If we come here, we need to grow the refcount table. Again, a new

     * refcount table needs some space and we can't simply allocate to avoid

     * endless recursion.

     *

     * Therefore let's grab new refcount blocks at the end of the image, which

     * will describe themselves and the new refcount table. This way we can

     * reference them only in the new table and do the switch to the new

     * refcount table at once without producing an inconsistent state in

     * between.

     */

    BLKDBG_EVENT(bs->file, BLKDBG_REFTABLE_GROW);



    /* Calculate the number of refcount blocks needed so far; this will be the

     * basis for calculating the index of the first cluster used for the

     * self-describing refcount structures which we are about to create.

     *

     * Because we reached this point, there cannot be any refcount entries for

     * cluster_index or higher indices yet. However, because new_block has been

     * allocated to describe that cluster (and it will assume this role later

     * on), we cannot use that index; also, new_block may actually have a higher

     * cluster index than cluster_index, so it needs to be taken into account

     * here (and 1 needs to be added to its value because that cluster is used).

     */

    uint64_t blocks_used = DIV_ROUND_UP(MAX(cluster_index + 1,

                                            (new_block >> s->cluster_bits) + 1),

                                        s->refcount_block_size);



    if (blocks_used > QCOW_MAX_REFTABLE_SIZE / sizeof(uint64_t)) {

        return -EFBIG;

    }



    /* And now we need at least one block more for the new metadata */

    uint64_t table_size = next_refcount_table_size(s, blocks_used + 1);

    uint64_t last_table_size;

    uint64_t blocks_clusters;

    do {

        uint64_t table_clusters =

            size_to_clusters(s, table_size * sizeof(uint64_t));

        blocks_clusters = 1 +

            ((table_clusters + s->refcount_block_size - 1)

            / s->refcount_block_size);

        uint64_t meta_clusters = table_clusters + blocks_clusters;



        last_table_size = table_size;

        table_size = next_refcount_table_size(s, blocks_used +

            ((meta_clusters + s->refcount_block_size - 1)

            / s->refcount_block_size));



    } while (last_table_size != table_size);



#ifdef DEBUG_ALLOC2

    fprintf(stderr, "qcow2: Grow refcount table %" PRId32 " => %" PRId64 "\n",

        s->refcount_table_size, table_size);

#endif



    /* Create the new refcount table and blocks */

    uint64_t meta_offset = (blocks_used * s->refcount_block_size) *

        s->cluster_size;

    uint64_t table_offset = meta_offset + blocks_clusters * s->cluster_size;

    uint64_t *new_table = g_try_new0(uint64_t, table_size);

    void *new_blocks = g_try_malloc0(blocks_clusters * s->cluster_size);



    assert(table_size > 0 && blocks_clusters > 0);

    if (new_table == NULL || new_blocks == NULL) {

        ret = -ENOMEM;

        goto fail_table;

    }



    /* Fill the new refcount table */

    memcpy(new_table, s->refcount_table,

        s->refcount_table_size * sizeof(uint64_t));

    new_table[refcount_table_index] = new_block;



    int i;

    for (i = 0; i < blocks_clusters; i++) {

        new_table[blocks_used + i] = meta_offset + (i * s->cluster_size);

    }



    /* Fill the refcount blocks */

    uint64_t table_clusters = size_to_clusters(s, table_size * sizeof(uint64_t));

    int block = 0;

    for (i = 0; i < table_clusters + blocks_clusters; i++) {

        s->set_refcount(new_blocks, block++, 1);

    }



    /* Write refcount blocks to disk */

    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_WRITE_BLOCKS);

    ret = bdrv_pwrite_sync(bs->file->bs, meta_offset, new_blocks,

        blocks_clusters * s->cluster_size);

    g_free(new_blocks);

    new_blocks = NULL;

    if (ret < 0) {

        goto fail_table;

    }



    /* Write refcount table to disk */

    for(i = 0; i < table_size; i++) {

        cpu_to_be64s(&new_table[i]);

    }



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_WRITE_TABLE);

    ret = bdrv_pwrite_sync(bs->file->bs, table_offset, new_table,

        table_size * sizeof(uint64_t));

    if (ret < 0) {

        goto fail_table;

    }



    for(i = 0; i < table_size; i++) {

        be64_to_cpus(&new_table[i]);

    }



    /* Hook up the new refcount table in the qcow2 header */

    uint8_t data[12];

    cpu_to_be64w((uint64_t*)data, table_offset);

    cpu_to_be32w((uint32_t*)(data + 8), table_clusters);

    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_SWITCH_TABLE);

    ret = bdrv_pwrite_sync(bs->file->bs,

                           offsetof(QCowHeader, refcount_table_offset),

                           data, sizeof(data));

    if (ret < 0) {

        goto fail_table;

    }



    /* And switch it in memory */

    uint64_t old_table_offset = s->refcount_table_offset;

    uint64_t old_table_size = s->refcount_table_size;



    g_free(s->refcount_table);

    s->refcount_table = new_table;

    s->refcount_table_size = table_size;

    s->refcount_table_offset = table_offset;



    /* Free old table. */

    qcow2_free_clusters(bs, old_table_offset, old_table_size * sizeof(uint64_t),

                        QCOW2_DISCARD_OTHER);



    ret = load_refcount_block(bs, new_block, refcount_block);

    if (ret < 0) {

        return ret;

    }



    /* If we were trying to do the initial refcount update for some cluster

     * allocation, we might have used the same clusters to store newly

     * allocated metadata. Make the caller search some new space. */

    return -EAGAIN;



fail_table:

    g_free(new_blocks);

    g_free(new_table);

fail_block:

    if (*refcount_block != NULL) {

        qcow2_cache_put(bs, s->refcount_block_cache, refcount_block);

    }

    return ret;

}
