static void xhci_kick_ep(XHCIState *xhci, unsigned int slotid,

                         unsigned int epid, unsigned int streamid)

{

    XHCIStreamContext *stctx;

    XHCIEPContext *epctx;

    XHCIRing *ring;

    USBEndpoint *ep = NULL;

    uint64_t mfindex;

    int length;

    int i;



    trace_usb_xhci_ep_kick(slotid, epid, streamid);

    assert(slotid >= 1 && slotid <= xhci->numslots);

    assert(epid >= 1 && epid <= 31);



    if (!xhci->slots[slotid-1].enabled) {

        fprintf(stderr, "xhci: xhci_kick_ep for disabled slot %d\n", slotid);

        return;

    }

    epctx = xhci->slots[slotid-1].eps[epid-1];

    if (!epctx) {

        fprintf(stderr, "xhci: xhci_kick_ep for disabled endpoint %d,%d\n",

                epid, slotid);

        return;

    }



    if (epctx->retry) {

        XHCITransfer *xfer = epctx->retry;



        trace_usb_xhci_xfer_retry(xfer);

        assert(xfer->running_retry);

        if (xfer->iso_xfer) {

            /* retry delayed iso transfer */

            mfindex = xhci_mfindex_get(xhci);

            xhci_check_iso_kick(xhci, xfer, epctx, mfindex);

            if (xfer->running_retry) {

                return;

            }

            if (xhci_setup_packet(xfer) < 0) {

                return;

            }

            usb_handle_packet(xfer->packet.ep->dev, &xfer->packet);

            assert(xfer->packet.status != USB_RET_NAK);

            xhci_complete_packet(xfer);

        } else {

            /* retry nak'ed transfer */

            if (xhci_setup_packet(xfer) < 0) {

                return;

            }

            usb_handle_packet(xfer->packet.ep->dev, &xfer->packet);

            if (xfer->packet.status == USB_RET_NAK) {

                return;

            }

            xhci_complete_packet(xfer);

        }

        assert(!xfer->running_retry);

        epctx->retry = NULL;

    }



    if (epctx->state == EP_HALTED) {

        DPRINTF("xhci: ep halted, not running schedule\n");

        return;

    }





    if (epctx->nr_pstreams) {

        uint32_t err;

        stctx = xhci_find_stream(epctx, streamid, &err);

        if (stctx == NULL) {

            return;

        }

        ring = &stctx->ring;

        xhci_set_ep_state(xhci, epctx, stctx, EP_RUNNING);

    } else {

        ring = &epctx->ring;

        streamid = 0;

        xhci_set_ep_state(xhci, epctx, NULL, EP_RUNNING);

    }

    assert(ring->dequeue != 0);



    while (1) {

        XHCITransfer *xfer = &epctx->transfers[epctx->next_xfer];

        if (xfer->running_async || xfer->running_retry) {

            break;

        }

        length = xhci_ring_chain_length(xhci, ring);

        if (length < 0) {

            break;

        } else if (length == 0) {

            break;

        }

        if (xfer->trbs && xfer->trb_alloced < length) {

            xfer->trb_count = 0;

            xfer->trb_alloced = 0;

            g_free(xfer->trbs);

            xfer->trbs = NULL;

        }

        if (!xfer->trbs) {

            xfer->trbs = g_malloc(sizeof(XHCITRB) * length);

            xfer->trb_alloced = length;

        }

        xfer->trb_count = length;



        for (i = 0; i < length; i++) {

            assert(xhci_ring_fetch(xhci, ring, &xfer->trbs[i], NULL));

        }

        xfer->xhci = xhci;

        xfer->epid = epid;

        xfer->slotid = slotid;

        xfer->streamid = streamid;



        if (epid == 1) {

            if (xhci_fire_ctl_transfer(xhci, xfer) >= 0) {

                epctx->next_xfer = (epctx->next_xfer + 1) % TD_QUEUE;

                ep = xfer->packet.ep;

            } else {

                fprintf(stderr, "xhci: error firing CTL transfer\n");

            }

        } else {

            if (xhci_fire_transfer(xhci, xfer, epctx) >= 0) {

                epctx->next_xfer = (epctx->next_xfer + 1) % TD_QUEUE;

                ep = xfer->packet.ep;

            } else {

                if (!xfer->iso_xfer) {

                    fprintf(stderr, "xhci: error firing data transfer\n");

                }

            }

        }



        if (epctx->state == EP_HALTED) {

            break;

        }

        if (xfer->running_retry) {

            DPRINTF("xhci: xfer nacked, stopping schedule\n");

            epctx->retry = xfer;

            break;

        }

    }

    if (ep) {

        usb_device_flush_ep_queue(ep->dev, ep);

    }

}
