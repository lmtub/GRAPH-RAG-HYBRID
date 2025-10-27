static AVBufferRef *pool_alloc_buffer(AVBufferPool *pool)

{

    BufferPoolEntry *buf;

    AVBufferRef     *ret;



    ret = pool->alloc(pool->size);

    if (!ret)

        return NULL;



    buf = av_mallocz(sizeof(*buf));

    if (!buf) {

        av_buffer_unref(&ret);

        return NULL;

    }



    buf->data   = ret->buffer->data;

    buf->opaque = ret->buffer->opaque;

    buf->free   = ret->buffer->free;

    buf->pool   = pool;



    ret->buffer->opaque = buf;

    ret->buffer->free   = pool_release_buffer;



    avpriv_atomic_int_add_and_fetch(&pool->refcount, 1);




    return ret;

}