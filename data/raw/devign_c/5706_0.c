static XHCIEPContext *xhci_alloc_epctx(XHCIState *xhci,

                                       unsigned int slotid,

                                       unsigned int epid)

{

    XHCIEPContext *epctx;

    int i;



    epctx = g_new0(XHCIEPContext, 1);

    epctx->xhci = xhci;

    epctx->slotid = slotid;

    epctx->epid = epid;



    for (i = 0; i < ARRAY_SIZE(epctx->transfers); i++) {

        epctx->transfers[i].xhci = xhci;

        epctx->transfers[i].slotid = slotid;

        epctx->transfers[i].epid = epid;

        usb_packet_init(&epctx->transfers[i].packet);

    }

    epctx->kick_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, xhci_ep_kick_timer, epctx);



    return epctx;

}
