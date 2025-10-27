ThreadPool *thread_pool_new(AioContext *ctx)

{

    ThreadPool *pool = g_new(ThreadPool, 1);

    thread_pool_init_one(pool, ctx);

    return pool;

}
