static TRBCCode xhci_disable_ep(XHCIState *xhci, unsigned int slotid,

                               unsigned int epid)

{

    XHCISlot *slot;

    XHCIEPContext *epctx;

    int i;



    trace_usb_xhci_ep_disable(slotid, epid);

    assert(slotid >= 1 && slotid <= xhci->numslots);

    assert(epid >= 1 && epid <= 31);



    slot = &xhci->slots[slotid-1];



    if (!slot->eps[epid-1]) {

        DPRINTF("xhci: slot %d ep %d already disabled\n", slotid, epid);

        return CC_SUCCESS;

    }



    xhci_ep_nuke_xfers(xhci, slotid, epid, 0);



    epctx = slot->eps[epid-1];



    if (epctx->nr_pstreams) {

        xhci_free_streams(epctx);

    }



    for (i = 0; i < ARRAY_SIZE(epctx->transfers); i++) {

        usb_packet_cleanup(&epctx->transfers[i].packet);

    }



    xhci_set_ep_state(xhci, epctx, NULL, EP_DISABLED);



    timer_free(epctx->kick_timer);

    g_free(epctx);

    slot->eps[epid-1] = NULL;



    return CC_SUCCESS;

}
