static void ehci_queues_rip_device(EHCIState *ehci, USBDevice *dev)

{

    EHCIQueue *q, *tmp;



    QTAILQ_FOREACH_SAFE(q, &ehci->queues, next, tmp) {

        if (q->packet.owner == NULL ||

            q->packet.owner->dev != dev) {

            continue;

        }

        ehci_free_queue(q);

    }

}
