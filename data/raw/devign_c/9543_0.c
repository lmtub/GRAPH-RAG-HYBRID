static void uhci_async_cancel_device(UHCIState *s, USBDevice *dev)

{

    UHCIQueue *queue;

    UHCIAsync *curr, *n;



    QTAILQ_FOREACH(queue, &s->queues, next) {

        QTAILQ_FOREACH_SAFE(curr, &queue->asyncs, next, n) {

            if (!usb_packet_is_inflight(&curr->packet) ||

                curr->packet.ep->dev != dev) {

                continue;

            }

            uhci_async_cancel(curr);

        }

    }

}
