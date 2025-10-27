static int xhci_ep_nuke_one_xfer(XHCITransfer *t, TRBCCode report)

{

    int killed = 0;



    if (report && (t->running_async || t->running_retry)) {

        t->status = report;

        xhci_xfer_report(t);

    }



    if (t->running_async) {

        usb_cancel_packet(&t->packet);

        t->running_async = 0;

        killed = 1;

    }

    if (t->running_retry) {

        XHCIEPContext *epctx = t->xhci->slots[t->slotid-1].eps[t->epid-1];

        if (epctx) {

            epctx->retry = NULL;

            timer_del(epctx->kick_timer);

        }

        t->running_retry = 0;

        killed = 1;

    }

    g_free(t->trbs);



    t->trbs = NULL;

    t->trb_count = t->trb_alloced = 0;



    return killed;

}
