static void uhci_async_validate_end(UHCIState *s)

{

    UHCIQueue *queue, *n;



    QTAILQ_FOREACH_SAFE(queue, &s->queues, next, n) {

        if (!queue->valid) {

            uhci_queue_free(queue);

        }

    }

}
