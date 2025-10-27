static AioContext *thread_pool_get_aio_context(BlockAIOCB *acb)

{

    ThreadPoolElement *elem = (ThreadPoolElement *)acb;

    ThreadPool *pool = elem->pool;

    return pool->ctx;

}
