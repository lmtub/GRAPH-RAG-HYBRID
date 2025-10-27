void thread_pool_free(ThreadPool *pool)

{

    if (!pool) {

        return;

    }



    assert(QLIST_EMPTY(&pool->head));



    qemu_mutex_lock(&pool->lock);



    /* Stop new threads from spawning */

    qemu_bh_delete(pool->new_thread_bh);

    pool->cur_threads -= pool->new_threads;

    pool->new_threads = 0;



    /* Wait for worker threads to terminate */

    pool->stopping = true;

    while (pool->cur_threads > 0) {

        qemu_sem_post(&pool->sem);

        qemu_cond_wait(&pool->worker_stopped, &pool->lock);

    }



    qemu_mutex_unlock(&pool->lock);



    qemu_bh_delete(pool->completion_bh);

    qemu_sem_destroy(&pool->sem);

    qemu_cond_destroy(&pool->worker_stopped);

    qemu_mutex_destroy(&pool->lock);

    g_free(pool);

}
