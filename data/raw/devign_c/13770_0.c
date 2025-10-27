static int ehci_state_writeback(EHCIQueue *q, int async)

{

    int again = 0;



    /*  Write back the QTD from the QH area */

    ehci_trace_qtd(q, NLPTR_GET(q->qtdaddr), (EHCIqtd*) &q->qh.next_qtd);

    put_dwords(NLPTR_GET(q->qtdaddr),(uint32_t *) &q->qh.next_qtd,

                sizeof(EHCIqtd) >> 2);



    /*

     * EHCI specs say go horizontal here.

     *

     * We can also advance the queue here for performance reasons.  We

     * need to take care to only take that shortcut in case we've

     * processed the qtd just written back without errors, i.e. halt

     * bit is clear.

     */

    if (q->qh.token & QTD_TOKEN_HALT) {

        ehci_set_state(q->ehci, async, EST_HORIZONTALQH);

        again = 1;

    } else {

        ehci_set_state(q->ehci, async, EST_ADVANCEQUEUE);

        again = 1;

    }

    return again;

}
