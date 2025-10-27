static int IRQ_get_next(OpenPICState *opp, IRQ_queue_t *q)

{

    if (q->next == -1) {

        /* XXX: optimize */

        IRQ_check(opp, q);

    }



    return q->next;

}
