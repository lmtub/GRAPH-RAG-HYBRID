static void usb_xhci_realize(struct PCIDevice *dev, Error **errp)

{

    int i, ret;

    Error *err = NULL;



    XHCIState *xhci = XHCI(dev);



    dev->config[PCI_CLASS_PROG] = 0x30;    /* xHCI */

    dev->config[PCI_INTERRUPT_PIN] = 0x01; /* interrupt pin 1 */

    dev->config[PCI_CACHE_LINE_SIZE] = 0x10;

    dev->config[0x60] = 0x30; /* release number */



    usb_xhci_init(xhci);



    if (xhci->msi != ON_OFF_AUTO_OFF) {

        ret = msi_init(dev, 0x70, xhci->numintrs, true, false, &err);

        /* Any error other than -ENOTSUP(board's MSI support is broken)

         * is a programming error */

        assert(!ret || ret == -ENOTSUP);

        if (ret && xhci->msi == ON_OFF_AUTO_ON) {

            /* Can't satisfy user's explicit msi=on request, fail */

            error_append_hint(&err, "You have to use msi=auto (default) or "

                    "msi=off with this machine type.\n");

            error_propagate(errp, err);

            return;

        }

        assert(!err || xhci->msi == ON_OFF_AUTO_AUTO);

        /* With msi=auto, we fall back to MSI off silently */

        error_free(err);

    }



    if (xhci->numintrs > MAXINTRS) {

        xhci->numintrs = MAXINTRS;

    }

    while (xhci->numintrs & (xhci->numintrs - 1)) {   /* ! power of 2 */

        xhci->numintrs++;

    }

    if (xhci->numintrs < 1) {

        xhci->numintrs = 1;

    }

    if (xhci->numslots > MAXSLOTS) {

        xhci->numslots = MAXSLOTS;

    }

    if (xhci->numslots < 1) {

        xhci->numslots = 1;

    }

    if (xhci_get_flag(xhci, XHCI_FLAG_ENABLE_STREAMS)) {

        xhci->max_pstreams_mask = 7; /* == 256 primary streams */

    } else {

        xhci->max_pstreams_mask = 0;

    }



    xhci->mfwrap_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, xhci_mfwrap_timer, xhci);



    memory_region_init(&xhci->mem, OBJECT(xhci), "xhci", LEN_REGS);

    memory_region_init_io(&xhci->mem_cap, OBJECT(xhci), &xhci_cap_ops, xhci,

                          "capabilities", LEN_CAP);

    memory_region_init_io(&xhci->mem_oper, OBJECT(xhci), &xhci_oper_ops, xhci,

                          "operational", 0x400);

    memory_region_init_io(&xhci->mem_runtime, OBJECT(xhci), &xhci_runtime_ops, xhci,

                          "runtime", LEN_RUNTIME);

    memory_region_init_io(&xhci->mem_doorbell, OBJECT(xhci), &xhci_doorbell_ops, xhci,

                          "doorbell", LEN_DOORBELL);



    memory_region_add_subregion(&xhci->mem, 0,            &xhci->mem_cap);

    memory_region_add_subregion(&xhci->mem, OFF_OPER,     &xhci->mem_oper);

    memory_region_add_subregion(&xhci->mem, OFF_RUNTIME,  &xhci->mem_runtime);

    memory_region_add_subregion(&xhci->mem, OFF_DOORBELL, &xhci->mem_doorbell);



    for (i = 0; i < xhci->numports; i++) {

        XHCIPort *port = &xhci->ports[i];

        uint32_t offset = OFF_OPER + 0x400 + 0x10 * i;

        port->xhci = xhci;

        memory_region_init_io(&port->mem, OBJECT(xhci), &xhci_port_ops, port,

                              port->name, 0x10);

        memory_region_add_subregion(&xhci->mem, offset, &port->mem);

    }



    pci_register_bar(dev, 0,

                     PCI_BASE_ADDRESS_SPACE_MEMORY|PCI_BASE_ADDRESS_MEM_TYPE_64,

                     &xhci->mem);



    if (pci_bus_is_express(dev->bus) ||

        xhci_get_flag(xhci, XHCI_FLAG_FORCE_PCIE_ENDCAP)) {

        ret = pcie_endpoint_cap_init(dev, 0xa0);

        assert(ret >= 0);

    }



    if (xhci->msix != ON_OFF_AUTO_OFF) {

        /* TODO check for errors */

        msix_init(dev, xhci->numintrs,

                  &xhci->mem, 0, OFF_MSIX_TABLE,

                  &xhci->mem, 0, OFF_MSIX_PBA,

                  0x90);

    }

}
