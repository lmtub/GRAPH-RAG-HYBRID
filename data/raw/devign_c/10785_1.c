static coroutine_fn int qcow2_co_flush_to_os(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    int ret;



    qemu_co_mutex_lock(&s->lock);

    ret = qcow2_cache_flush(bs, s->l2_table_cache);

    if (ret < 0) {

        qemu_co_mutex_unlock(&s->lock);

        return ret;

    }



    ret = qcow2_cache_flush(bs, s->refcount_block_cache);

    if (ret < 0) {

        qemu_co_mutex_unlock(&s->lock);

        return ret;

    }

    qemu_co_mutex_unlock(&s->lock);



    return 0;

}
