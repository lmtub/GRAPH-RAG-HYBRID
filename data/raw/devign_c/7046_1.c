static void ehci_queues_rip_unused(EHCIState *ehci, int async)

{

    EHCIQueueHead *head = async ? &ehci->aqueues : &ehci->pqueues;

    EHCIQueue *q, *tmp;



    QTAILQ_FOREACH_SAFE(q, head, next, tmp) {

        if (q->seen) {

            q->seen = 0;

            q->ts = ehci->last_run_ns;

            continue;

        }

        if (ehci->last_run_ns < q->ts + 250000000) {

            /* allow 0.25 sec idle */

            continue;

        }

        ehci_free_queue(q, async);

    }

}
