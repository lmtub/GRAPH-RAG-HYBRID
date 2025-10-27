static void xhci_xfer_report(XHCITransfer *xfer)

{

    uint32_t edtla = 0;

    unsigned int left;

    bool reported = 0;

    bool shortpkt = 0;

    XHCIEvent event = {ER_TRANSFER, CC_SUCCESS};

    XHCIState *xhci = xfer->xhci;

    int i;



    left = xfer->packet.actual_length;



    for (i = 0; i < xfer->trb_count; i++) {

        XHCITRB *trb = &xfer->trbs[i];

        unsigned int chunk = 0;



        switch (TRB_TYPE(*trb)) {

        case TR_DATA:

        case TR_NORMAL:

        case TR_ISOCH:

            chunk = trb->status & 0x1ffff;

            if (chunk > left) {

                chunk = left;

                if (xfer->status == CC_SUCCESS) {

                    shortpkt = 1;

                }

            }

            left -= chunk;

            edtla += chunk;

            break;

        case TR_STATUS:

            reported = 0;

            shortpkt = 0;

            break;

        }



        if (!reported && ((trb->control & TRB_TR_IOC) ||

                          (shortpkt && (trb->control & TRB_TR_ISP)) ||

                          (xfer->status != CC_SUCCESS && left == 0))) {

            event.slotid = xfer->slotid;

            event.epid = xfer->epid;

            event.length = (trb->status & 0x1ffff) - chunk;

            event.flags = 0;

            event.ptr = trb->addr;

            if (xfer->status == CC_SUCCESS) {

                event.ccode = shortpkt ? CC_SHORT_PACKET : CC_SUCCESS;

            } else {

                event.ccode = xfer->status;

            }

            if (TRB_TYPE(*trb) == TR_EVDATA) {

                event.ptr = trb->parameter;

                event.flags |= TRB_EV_ED;

                event.length = edtla & 0xffffff;

                DPRINTF("xhci_xfer_data: EDTLA=%d\n", event.length);

                edtla = 0;

            }

            xhci_event(xhci, &event, TRB_INTR(*trb));

            reported = 1;

            if (xfer->status != CC_SUCCESS) {

                return;

            }

        }

    }

}
