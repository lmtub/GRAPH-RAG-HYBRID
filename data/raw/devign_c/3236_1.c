static void xhci_reset(DeviceState *dev)

{

    XHCIState *xhci = XHCI(dev);

    int i;



    trace_usb_xhci_reset();

    if (!(xhci->usbsts & USBSTS_HCH)) {

        DPRINTF("xhci: reset while running!\n");

    }



    xhci->usbcmd = 0;

    xhci->usbsts = USBSTS_HCH;

    xhci->dnctrl = 0;

    xhci->crcr_low = 0;

    xhci->crcr_high = 0;

    xhci->dcbaap_low = 0;

    xhci->dcbaap_high = 0;

    xhci->config = 0;



    for (i = 0; i < xhci->numslots; i++) {

        xhci_disable_slot(xhci, i+1);

    }



    for (i = 0; i < xhci->numports; i++) {

        xhci_port_update(xhci->ports + i, 0);

    }



    for (i = 0; i < xhci->numintrs; i++) {

        xhci->intr[i].iman = 0;

        xhci->intr[i].imod = 0;

        xhci->intr[i].erstsz = 0;

        xhci->intr[i].erstba_low = 0;

        xhci->intr[i].erstba_high = 0;

        xhci->intr[i].erdp_low = 0;

        xhci->intr[i].erdp_high = 0;

        xhci->intr[i].msix_used = 0;



        xhci->intr[i].er_ep_idx = 0;

        xhci->intr[i].er_pcs = 1;

        xhci->intr[i].er_full = 0;

        xhci->intr[i].ev_buffer_put = 0;

        xhci->intr[i].ev_buffer_get = 0;

    }



    xhci->mfindex_start = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    xhci_mfwrap_update(xhci);

}
