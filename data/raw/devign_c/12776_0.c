static void spawn_thread_bh_fn(void *opaque)

{

    ThreadPool *pool = opaque;



    qemu_mutex_lock(&pool->lock);

    do_spawn_thread(pool);

    qemu_mutex_unlock(&pool->lock);

}
