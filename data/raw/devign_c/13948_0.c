static void *worker_thread(void *opaque)

{

    ThreadPool *pool = opaque;



    qemu_mutex_lock(&pool->lock);

    pool->pending_threads--;

    do_spawn_thread(pool);



    while (!pool->stopping) {

        ThreadPoolElement *req;

        int ret;



        do {

            pool->idle_threads++;

            qemu_mutex_unlock(&pool->lock);

            ret = qemu_sem_timedwait(&pool->sem, 10000);

            qemu_mutex_lock(&pool->lock);

            pool->idle_threads--;

        } while (ret == -1 && !QTAILQ_EMPTY(&pool->request_list));

        if (ret == -1 || pool->stopping) {

            break;

        }



        req = QTAILQ_FIRST(&pool->request_list);

        QTAILQ_REMOVE(&pool->request_list, req, reqs);

        req->state = THREAD_ACTIVE;

        qemu_mutex_unlock(&pool->lock);



        ret = req->func(req->arg);



        req->ret = ret;

        /* Write ret before state.  */

        smp_wmb();

        req->state = THREAD_DONE;



        qemu_mutex_lock(&pool->lock);



        qemu_bh_schedule(pool->completion_bh);

    }



    pool->cur_threads--;

    qemu_cond_signal(&pool->worker_stopped);

    qemu_mutex_unlock(&pool->lock);

    return NULL;

}
