static EHCIQueue *ehci_state_fetchqh(EHCIState *ehci, int async)

{

    uint32_t entry;

    EHCIQueue *q;

    int reload;



    entry = ehci_get_fetch_addr(ehci, async);

    q = ehci_find_queue_by_qh(ehci, entry);

    if (NULL == q) {

        q = ehci_alloc_queue(ehci, async);

    }

    q->qhaddr = entry;

    q->seen++;



    if (q->seen > 1) {

        /* we are going in circles -- stop processing */

        ehci_set_state(ehci, async, EST_ACTIVE);

        q = NULL;

        goto out;

    }



    get_dwords(NLPTR_GET(q->qhaddr), (uint32_t *) &q->qh, sizeof(EHCIqh) >> 2);

    ehci_trace_qh(q, NLPTR_GET(q->qhaddr), &q->qh);



    if (q->async == EHCI_ASYNC_INFLIGHT) {

        /* I/O still in progress -- skip queue */

        ehci_set_state(ehci, async, EST_HORIZONTALQH);

        goto out;

    }

    if (q->async == EHCI_ASYNC_FINISHED) {

        /* I/O finished -- continue processing queue */

        trace_usb_ehci_queue_action(q, "resume");

        ehci_set_state(ehci, async, EST_EXECUTING);

        goto out;

    }



    if (async && (q->qh.epchar & QH_EPCHAR_H)) {



        /*  EHCI spec version 1.0 Section 4.8.3 & 4.10.1 */

        if (ehci->usbsts & USBSTS_REC) {

            ehci_clear_usbsts(ehci, USBSTS_REC);

        } else {

            DPRINTF("FETCHQH:  QH 0x%08x. H-bit set, reclamation status reset"

                       " - done processing\n", q->qhaddr);

            ehci_set_state(ehci, async, EST_ACTIVE);

            q = NULL;

            goto out;

        }

    }



#if EHCI_DEBUG

    if (q->qhaddr != q->qh.next) {

    DPRINTF("FETCHQH:  QH 0x%08x (h %x halt %x active %x) next 0x%08x\n",

               q->qhaddr,

               q->qh.epchar & QH_EPCHAR_H,

               q->qh.token & QTD_TOKEN_HALT,

               q->qh.token & QTD_TOKEN_ACTIVE,

               q->qh.next);

    }

#endif



    reload = get_field(q->qh.epchar, QH_EPCHAR_RL);

    if (reload) {

        set_field(&q->qh.altnext_qtd, reload, QH_ALTNEXT_NAKCNT);

    }



    if (q->qh.token & QTD_TOKEN_HALT) {

        ehci_set_state(ehci, async, EST_HORIZONTALQH);



    } else if ((q->qh.token & QTD_TOKEN_ACTIVE) && (q->qh.current_qtd > 0x1000)) {

        q->qtdaddr = q->qh.current_qtd;

        ehci_set_state(ehci, async, EST_FETCHQTD);



    } else {

        /*  EHCI spec version 1.0 Section 4.10.2 */

        ehci_set_state(ehci, async, EST_ADVANCEQUEUE);

    }



out:

    return q;

}
