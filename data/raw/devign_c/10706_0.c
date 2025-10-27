static int ehci_state_fetchqtd(EHCIQueue *q, int async)

{

    int again = 0;



    get_dwords(NLPTR_GET(q->qtdaddr),(uint32_t *) &q->qtd, sizeof(EHCIqtd) >> 2);

    ehci_trace_qtd(q, NLPTR_GET(q->qtdaddr), &q->qtd);



    if (q->qtd.token & QTD_TOKEN_ACTIVE) {

        ehci_set_state(q->ehci, async, EST_EXECUTE);

        again = 1;

    } else {

        ehci_set_state(q->ehci, async, EST_HORIZONTALQH);

        again = 1;

    }



    return again;

}
