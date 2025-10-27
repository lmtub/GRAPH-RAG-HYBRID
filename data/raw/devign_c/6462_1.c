int qcow2_get_refcount(BlockDriverState *bs, int64_t cluster_index,

                       uint64_t *refcount)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t refcount_table_index, block_index;

    int64_t refcount_block_offset;

    int ret;

    uint16_t *refcount_block;



    refcount_table_index = cluster_index >> s->refcount_block_bits;

    if (refcount_table_index >= s->refcount_table_size) {

        *refcount = 0;

        return 0;

    }

    refcount_block_offset =

        s->refcount_table[refcount_table_index] & REFT_OFFSET_MASK;

    if (!refcount_block_offset) {

        *refcount = 0;

        return 0;

    }



    if (offset_into_cluster(s, refcount_block_offset)) {

        qcow2_signal_corruption(bs, true, -1, -1, "Refblock offset %#" PRIx64

                                " unaligned (reftable index: %#" PRIx64 ")",

                                refcount_block_offset, refcount_table_index);

        return -EIO;

    }



    ret = qcow2_cache_get(bs, s->refcount_block_cache, refcount_block_offset,

        (void**) &refcount_block);

    if (ret < 0) {

        return ret;

    }



    block_index = cluster_index & (s->refcount_block_size - 1);

    *refcount = be16_to_cpu(refcount_block[block_index]);



    ret = qcow2_cache_put(bs, s->refcount_block_cache,

        (void**) &refcount_block);

    if (ret < 0) {

        return ret;

    }



    return 0;

}
