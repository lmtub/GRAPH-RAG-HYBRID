static int write_refcount_block(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    size_t size = s->cluster_size;



    if (s->refcount_block_cache_offset == 0) {

        return 0;

    }



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_UPDATE);

    if (bdrv_pwrite(bs->file, s->refcount_block_cache_offset,

            s->refcount_block_cache, size) != size)

    {

        return -EIO;

    }



    return 0;

}
