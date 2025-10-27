static void ehci_queues_rip_unused(EHCIState *ehci, int async, int flush)

{

    EHCIQueueHead *head = async ? &ehci->aqueues : &ehci->pqueues;

    uint64_t maxage = FRAME_TIMER_NS * ehci->maxframes * 4;

    EHCIQueue *q, *tmp;



    QTAILQ_FOREACH_SAFE(q, head, next, tmp) {

        if (q->seen) {

            q->seen = 0;

            q->ts = ehci->last_run_ns;

            continue;

        }

        if (!flush && ehci->last_run_ns < q->ts + maxage) {

            continue;

        }

        ehci_free_queue(q);

    }

}
