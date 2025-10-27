static void uhci_async_cancel_all(UHCIState *s)

{

    UHCIQueue *queue;

    UHCIAsync *curr, *n;



    QTAILQ_FOREACH(queue, &s->queues, next) {

        QTAILQ_FOREACH_SAFE(curr, &queue->asyncs, next, n) {

            uhci_async_unlink(curr);

            uhci_async_cancel(curr);

        }

        uhci_queue_free(queue);

    }

}
