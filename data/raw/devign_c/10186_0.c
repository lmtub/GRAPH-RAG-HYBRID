static int xhci_submit(XHCIState *xhci, XHCITransfer *xfer, XHCIEPContext *epctx)

{

    uint64_t mfindex;



    DPRINTF("xhci_submit(slotid=%d,epid=%d)\n", xfer->slotid, xfer->epid);



    xfer->in_xfer = epctx->type>>2;



    switch(epctx->type) {

    case ET_INTR_OUT:

    case ET_INTR_IN:

        xfer->pkts = 0;

        xfer->iso_xfer = false;

        xfer->timed_xfer = true;

        mfindex = xhci_mfindex_get(xhci);

        xhci_calc_intr_kick(xhci, xfer, epctx, mfindex);

        xhci_check_intr_iso_kick(xhci, xfer, epctx, mfindex);

        if (xfer->running_retry) {

            return -1;

        }

        break;

    case ET_BULK_OUT:

    case ET_BULK_IN:

        xfer->pkts = 0;

        xfer->iso_xfer = false;

        xfer->timed_xfer = false;

        break;

    case ET_ISO_OUT:

    case ET_ISO_IN:

        xfer->pkts = 1;

        xfer->iso_xfer = true;

        xfer->timed_xfer = true;

        mfindex = xhci_mfindex_get(xhci);

        xhci_calc_iso_kick(xhci, xfer, epctx, mfindex);

        xhci_check_intr_iso_kick(xhci, xfer, epctx, mfindex);

        if (xfer->running_retry) {

            return -1;

        }

        break;

    default:

        trace_usb_xhci_unimplemented("endpoint type", epctx->type);

        return -1;

    }



    if (xhci_setup_packet(xfer) < 0) {

        return -1;

    }

    usb_handle_packet(xfer->packet.ep->dev, &xfer->packet);



    xhci_try_complete_packet(xfer);

    if (!xfer->running_async && !xfer->running_retry) {

        xhci_kick_epctx(xfer->epctx, xfer->streamid);

    }

    return 0;

}
