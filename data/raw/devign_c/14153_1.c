static int xhci_fire_ctl_transfer(XHCIState *xhci, XHCITransfer *xfer)

{

    XHCITRB *trb_setup, *trb_status;

    uint8_t bmRequestType, bRequest;

    uint16_t wValue, wLength, wIndex;

    XHCIPort *port;

    USBDevice *dev;

    int ret;



    DPRINTF("xhci_fire_ctl_transfer(slot=%d)\n", xfer->slotid);



    trb_setup = &xfer->trbs[0];

    trb_status = &xfer->trbs[xfer->trb_count-1];



    /* at most one Event Data TRB allowed after STATUS */

    if (TRB_TYPE(*trb_status) == TR_EVDATA && xfer->trb_count > 2) {

        trb_status--;

    }



    /* do some sanity checks */

    if (TRB_TYPE(*trb_setup) != TR_SETUP) {

        fprintf(stderr, "xhci: ep0 first TD not SETUP: %d\n",

                TRB_TYPE(*trb_setup));

        return -1;

    }

    if (TRB_TYPE(*trb_status) != TR_STATUS) {

        fprintf(stderr, "xhci: ep0 last TD not STATUS: %d\n",

                TRB_TYPE(*trb_status));

        return -1;

    }

    if (!(trb_setup->control & TRB_TR_IDT)) {

        fprintf(stderr, "xhci: Setup TRB doesn't have IDT set\n");

        return -1;

    }

    if ((trb_setup->status & 0x1ffff) != 8) {

        fprintf(stderr, "xhci: Setup TRB has bad length (%d)\n",

                (trb_setup->status & 0x1ffff));

        return -1;

    }



    bmRequestType = trb_setup->parameter;

    bRequest = trb_setup->parameter >> 8;

    wValue = trb_setup->parameter >> 16;

    wIndex = trb_setup->parameter >> 32;

    wLength = trb_setup->parameter >> 48;



    if (xfer->data && xfer->data_alloced < wLength) {

        xfer->data_alloced = 0;

        g_free(xfer->data);

        xfer->data = NULL;

    }

    if (!xfer->data) {

        DPRINTF("xhci: alloc %d bytes data\n", wLength);

        xfer->data = g_malloc(wLength+1);

        xfer->data_alloced = wLength;

    }

    xfer->data_length = wLength;



    port = &xhci->ports[xhci->slots[xfer->slotid-1].port-1];

    dev = xhci_find_device(port, xhci->slots[xfer->slotid-1].devaddr);

    if (!dev) {

        fprintf(stderr, "xhci: slot %d port %d has no device\n", xfer->slotid,

                xhci->slots[xfer->slotid-1].port);

        return -1;

    }



    xfer->in_xfer = bmRequestType & USB_DIR_IN;

    xfer->iso_xfer = false;



    xhci_setup_packet(xfer, dev);

    if (!xfer->in_xfer) {

        xhci_xfer_data(xfer, xfer->data, wLength, 0, 1, 0);

    }

    ret = usb_device_handle_control(dev, &xfer->packet,

                                    (bmRequestType << 8) | bRequest,

                                    wValue, wIndex, wLength, xfer->data);



    xhci_complete_packet(xfer, ret);

    if (!xfer->running_async && !xfer->running_retry) {

        xhci_kick_ep(xhci, xfer->slotid, xfer->epid);

    }

    return 0;

}
