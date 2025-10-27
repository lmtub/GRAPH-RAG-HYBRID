static void thread_pool_cancel(BlockDriverAIOCB *acb)

{

    ThreadPoolElement *elem = (ThreadPoolElement *)acb;

    ThreadPool *pool = elem->pool;



    trace_thread_pool_cancel(elem, elem->common.opaque);



    qemu_mutex_lock(&pool->lock);

    if (elem->state == THREAD_QUEUED &&

        /* No thread has yet started working on elem. we can try to "steal"

         * the item from the worker if we can get a signal from the

         * semaphore.  Because this is non-blocking, we can do it with

         * the lock taken and ensure that elem will remain THREAD_QUEUED.

         */

        qemu_sem_timedwait(&pool->sem, 0) == 0) {

        QTAILQ_REMOVE(&pool->request_list, elem, reqs);

        elem->state = THREAD_CANCELED;

        event_notifier_set(&pool->notifier);

    } else {

        pool->pending_cancellations++;

        while (elem->state != THREAD_CANCELED && elem->state != THREAD_DONE) {

            qemu_cond_wait(&pool->check_cancel, &pool->lock);

        }

        pool->pending_cancellations--;

    }

    qemu_mutex_unlock(&pool->lock);

    event_notifier_ready(&pool->notifier);

}
