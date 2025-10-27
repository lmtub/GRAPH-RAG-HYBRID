static TRBCCode xhci_address_slot(XHCIState *xhci, unsigned int slotid,

                                  uint64_t pictx, bool bsr)

{

    XHCISlot *slot;

    USBPort *uport;

    USBDevice *dev;

    dma_addr_t ictx, octx, dcbaap;

    uint64_t poctx;

    uint32_t ictl_ctx[2];

    uint32_t slot_ctx[4];

    uint32_t ep0_ctx[5];

    int i;

    TRBCCode res;



    assert(slotid >= 1 && slotid <= xhci->numslots);



    dcbaap = xhci_addr64(xhci->dcbaap_low, xhci->dcbaap_high);

    poctx = ldq_le_pci_dma(PCI_DEVICE(xhci), dcbaap + 8 * slotid);

    ictx = xhci_mask64(pictx);

    octx = xhci_mask64(poctx);



    DPRINTF("xhci: input context at "DMA_ADDR_FMT"\n", ictx);

    DPRINTF("xhci: output context at "DMA_ADDR_FMT"\n", octx);



    xhci_dma_read_u32s(xhci, ictx, ictl_ctx, sizeof(ictl_ctx));



    if (ictl_ctx[0] != 0x0 || ictl_ctx[1] != 0x3) {

        fprintf(stderr, "xhci: invalid input context control %08x %08x\n",

                ictl_ctx[0], ictl_ctx[1]);

        return CC_TRB_ERROR;

    }



    xhci_dma_read_u32s(xhci, ictx+32, slot_ctx, sizeof(slot_ctx));

    xhci_dma_read_u32s(xhci, ictx+64, ep0_ctx, sizeof(ep0_ctx));



    DPRINTF("xhci: input slot context: %08x %08x %08x %08x\n",

            slot_ctx[0], slot_ctx[1], slot_ctx[2], slot_ctx[3]);



    DPRINTF("xhci: input ep0 context: %08x %08x %08x %08x %08x\n",

            ep0_ctx[0], ep0_ctx[1], ep0_ctx[2], ep0_ctx[3], ep0_ctx[4]);



    uport = xhci_lookup_uport(xhci, slot_ctx);

    if (uport == NULL) {

        fprintf(stderr, "xhci: port not found\n");

        return CC_TRB_ERROR;

    }

    trace_usb_xhci_slot_address(slotid, uport->path);



    dev = uport->dev;

    if (!dev) {

        fprintf(stderr, "xhci: port %s not connected\n", uport->path);

        return CC_USB_TRANSACTION_ERROR;

    }



    for (i = 0; i < xhci->numslots; i++) {

        if (i == slotid-1) {

            continue;

        }

        if (xhci->slots[i].uport == uport) {

            fprintf(stderr, "xhci: port %s already assigned to slot %d\n",

                    uport->path, i+1);

            return CC_TRB_ERROR;

        }

    }



    slot = &xhci->slots[slotid-1];

    slot->uport = uport;

    slot->ctx = octx;



    if (bsr) {

        slot_ctx[3] = SLOT_DEFAULT << SLOT_STATE_SHIFT;

    } else {

        USBPacket p;

        uint8_t buf[1];



        slot_ctx[3] = (SLOT_ADDRESSED << SLOT_STATE_SHIFT) | slotid;

        usb_device_reset(dev);

        memset(&p, 0, sizeof(p));

        usb_packet_addbuf(&p, buf, sizeof(buf));

        usb_packet_setup(&p, USB_TOKEN_OUT,

                         usb_ep_get(dev, USB_TOKEN_OUT, 0), 0,

                         0, false, false);

        usb_device_handle_control(dev, &p,

                                  DeviceOutRequest | USB_REQ_SET_ADDRESS,

                                  slotid, 0, 0, NULL);

        assert(p.status != USB_RET_ASYNC);

    }



    res = xhci_enable_ep(xhci, slotid, 1, octx+32, ep0_ctx);



    DPRINTF("xhci: output slot context: %08x %08x %08x %08x\n",

            slot_ctx[0], slot_ctx[1], slot_ctx[2], slot_ctx[3]);

    DPRINTF("xhci: output ep0 context: %08x %08x %08x %08x %08x\n",

            ep0_ctx[0], ep0_ctx[1], ep0_ctx[2], ep0_ctx[3], ep0_ctx[4]);



    xhci_dma_write_u32s(xhci, octx, slot_ctx, sizeof(slot_ctx));

    xhci_dma_write_u32s(xhci, octx+32, ep0_ctx, sizeof(ep0_ctx));



    xhci->slots[slotid-1].addressed = 1;

    return res;

}
