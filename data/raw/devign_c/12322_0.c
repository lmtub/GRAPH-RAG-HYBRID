static int qcow2_discard_refcount_block(BlockDriverState *bs,

                                        uint64_t discard_block_offs)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t refblock_offs = get_refblock_offset(s, discard_block_offs);

    uint64_t cluster_index = discard_block_offs >> s->cluster_bits;

    uint32_t block_index = cluster_index & (s->refcount_block_size - 1);

    void *refblock;

    int ret;



    assert(discard_block_offs != 0);



    ret = qcow2_cache_get(bs, s->refcount_block_cache, refblock_offs,

                          &refblock);

    if (ret < 0) {

        return ret;

    }



    if (s->get_refcount(refblock, block_index) != 1) {

        qcow2_signal_corruption(bs, true, -1, -1, "Invalid refcount:"

                                " refblock offset %#" PRIx64

                                ", reftable index %u"

                                ", block offset %#" PRIx64

                                ", refcount %#" PRIx64,

                                refblock_offs,

                                offset_to_reftable_index(s, discard_block_offs),

                                discard_block_offs,

                                s->get_refcount(refblock, block_index));

        qcow2_cache_put(bs, s->refcount_block_cache, &refblock);

        return -EINVAL;

    }

    s->set_refcount(refblock, block_index, 0);



    qcow2_cache_entry_mark_dirty(bs, s->refcount_block_cache, refblock);



    qcow2_cache_put(bs, s->refcount_block_cache, &refblock);



    if (cluster_index < s->free_cluster_index) {

        s->free_cluster_index = cluster_index;

    }



    refblock = qcow2_cache_is_table_offset(bs, s->refcount_block_cache,

                                           discard_block_offs);

    if (refblock) {

        /* discard refblock from the cache if refblock is cached */

        qcow2_cache_discard(bs, s->refcount_block_cache, refblock);

    }

    update_refcount_discard(bs, discard_block_offs, s->cluster_size);



    return 0;

}
