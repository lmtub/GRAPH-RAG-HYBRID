static int ehci_qh_do_overlay(EHCIQueue *q)

{

    int i;

    int dtoggle;

    int ping;

    int eps;

    int reload;



    // remember values in fields to preserve in qh after overlay



    dtoggle = q->qh.token & QTD_TOKEN_DTOGGLE;

    ping    = q->qh.token & QTD_TOKEN_PING;



    q->qh.current_qtd = q->qtdaddr;

    q->qh.next_qtd    = q->qtd.next;

    q->qh.altnext_qtd = q->qtd.altnext;

    q->qh.token       = q->qtd.token;





    eps = get_field(q->qh.epchar, QH_EPCHAR_EPS);

    if (eps == EHCI_QH_EPS_HIGH) {

        q->qh.token &= ~QTD_TOKEN_PING;

        q->qh.token |= ping;

    }



    reload = get_field(q->qh.epchar, QH_EPCHAR_RL);

    set_field(&q->qh.altnext_qtd, reload, QH_ALTNEXT_NAKCNT);



    for (i = 0; i < 5; i++) {

        q->qh.bufptr[i] = q->qtd.bufptr[i];

    }



    if (!(q->qh.epchar & QH_EPCHAR_DTC)) {

        // preserve QH DT bit

        q->qh.token &= ~QTD_TOKEN_DTOGGLE;

        q->qh.token |= dtoggle;

    }



    q->qh.bufptr[1] &= ~BUFPTR_CPROGMASK_MASK;

    q->qh.bufptr[2] &= ~BUFPTR_FRAMETAG_MASK;



    put_dwords(NLPTR_GET(q->qhaddr), (uint32_t *) &q->qh, sizeof(EHCIqh) >> 2);



    return 0;

}
