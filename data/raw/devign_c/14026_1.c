static int ehci_state_fetchqtd(EHCIQueue *q)

{

    EHCIqtd qtd;

    EHCIPacket *p;

    int again = 0;



    get_dwords(q->ehci, NLPTR_GET(q->qtdaddr), (uint32_t *) &qtd,

               sizeof(EHCIqtd) >> 2);

    ehci_trace_qtd(q, NLPTR_GET(q->qtdaddr), &qtd);



    p = QTAILQ_FIRST(&q->packets);

    if (p != NULL) {

        if (p->qtdaddr != q->qtdaddr ||

            (!NLPTR_TBIT(p->qtd.next) && (p->qtd.next != qtd.next)) ||

            (!NLPTR_TBIT(p->qtd.altnext) && (p->qtd.altnext != qtd.altnext)) ||

            p->qtd.bufptr[0] != qtd.bufptr[0]) {

            ehci_cancel_queue(q);

            ehci_trace_guest_bug(q->ehci, "guest updated active QH or qTD");

            p = NULL;

        } else {

            p->qtd = qtd;

            ehci_qh_do_overlay(q);

        }

    }



    if (!(qtd.token & QTD_TOKEN_ACTIVE)) {

        if (p != NULL) {

            /* transfer canceled by guest (clear active) */

            ehci_cancel_queue(q);

            p = NULL;

        }

        ehci_set_state(q->ehci, q->async, EST_HORIZONTALQH);

        again = 1;

    } else if (p != NULL) {

        switch (p->async) {

        case EHCI_ASYNC_NONE:

            /* Previously nacked packet (likely interrupt ep) */

           ehci_set_state(q->ehci, q->async, EST_EXECUTE);

           break;

        case EHCI_ASYNC_INFLIGHT:

            /* Unfinyshed async handled packet, go horizontal */

            ehci_set_state(q->ehci, q->async, EST_HORIZONTALQH);

            break;

        case EHCI_ASYNC_FINISHED:

            /* Should never happen, as this case is caught by fetchqh */

            ehci_set_state(q->ehci, q->async, EST_EXECUTING);

            break;

        }

        again = 1;

    } else {

        p = ehci_alloc_packet(q);

        p->qtdaddr = q->qtdaddr;

        p->qtd = qtd;

        ehci_set_state(q->ehci, q->async, EST_EXECUTE);

        again = 1;

    }



    return again;

}
