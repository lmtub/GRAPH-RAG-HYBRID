static int qcow2_co_flush(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    int ret;



    qemu_co_mutex_lock(&s->lock);

    ret = qcow2_cache_flush(bs, s->l2_table_cache);

    if (ret < 0) {


        return ret;

    }



    ret = qcow2_cache_flush(bs, s->refcount_block_cache);

    if (ret < 0) {


        return ret;

    }




    return bdrv_co_flush(bs->file);

}