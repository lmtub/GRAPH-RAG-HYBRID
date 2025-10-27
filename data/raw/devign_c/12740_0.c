static void event_notifier_ready(EventNotifier *notifier)

{

    ThreadPool *pool = container_of(notifier, ThreadPool, notifier);

    ThreadPoolElement *elem, *next;



    event_notifier_test_and_clear(notifier);

restart:

    QLIST_FOREACH_SAFE(elem, &pool->head, all, next) {

        if (elem->state != THREAD_CANCELED && elem->state != THREAD_DONE) {

            continue;

        }

        if (elem->state == THREAD_DONE) {

            trace_thread_pool_complete(pool, elem, elem->common.opaque,

                                       elem->ret);

        }

        if (elem->state == THREAD_DONE && elem->common.cb) {

            QLIST_REMOVE(elem, all);

            /* Read state before ret.  */

            smp_rmb();

            elem->common.cb(elem->common.opaque, elem->ret);

            qemu_aio_release(elem);

            goto restart;

        } else {

            /* remove the request */

            QLIST_REMOVE(elem, all);

            qemu_aio_release(elem);

        }

    }

}
