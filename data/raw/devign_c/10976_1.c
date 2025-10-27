static int usb_xhci_post_load(void *opaque, int version_id)

{

    XHCIState *xhci = opaque;

    PCIDevice *pci_dev = PCI_DEVICE(xhci);

    XHCISlot *slot;

    XHCIEPContext *epctx;

    dma_addr_t dcbaap, pctx;

    uint32_t slot_ctx[4];

    uint32_t ep_ctx[5];

    int slotid, epid, state, intr;



    dcbaap = xhci_addr64(xhci->dcbaap_low, xhci->dcbaap_high);



    for (slotid = 1; slotid <= xhci->numslots; slotid++) {

        slot = &xhci->slots[slotid-1];

        if (!slot->addressed) {



        slot->ctx =

            xhci_mask64(ldq_le_pci_dma(pci_dev, dcbaap + 8 * slotid));

        xhci_dma_read_u32s(xhci, slot->ctx, slot_ctx, sizeof(slot_ctx));

        slot->uport = xhci_lookup_uport(xhci, slot_ctx);







        assert(slot->uport && slot->uport->dev);



        for (epid = 1; epid <= 31; epid++) {

            pctx = slot->ctx + 32 * epid;

            xhci_dma_read_u32s(xhci, pctx, ep_ctx, sizeof(ep_ctx));

            state = ep_ctx[0] & EP_STATE_MASK;

            if (state == EP_DISABLED) {



            epctx = xhci_alloc_epctx(xhci, slotid, epid);

            slot->eps[epid-1] = epctx;

            xhci_init_epctx(epctx, pctx, ep_ctx);

            epctx->state = state;

            if (state == EP_RUNNING) {

                /* kick endpoint after vmload is finished */

                timer_mod(epctx->kick_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL));






    for (intr = 0; intr < xhci->numintrs; intr++) {

        if (xhci->intr[intr].msix_used) {

            msix_vector_use(pci_dev, intr);

        } else {

            msix_vector_unuse(pci_dev, intr);





    return 0;
