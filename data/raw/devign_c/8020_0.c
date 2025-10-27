static char *pxb_host_ofw_unit_address(const SysBusDevice *dev)

{

    const PCIHostState *pxb_host;

    const PCIBus *pxb_bus;

    const PXBDev *pxb_dev;

    int position;

    const DeviceState *pxb_dev_base;

    const PCIHostState *main_host;

    const SysBusDevice *main_host_sbd;



    pxb_host = PCI_HOST_BRIDGE(dev);

    pxb_bus = pxb_host->bus;

    pxb_dev = PXB_DEV(pxb_bus->parent_dev);

    position = g_list_index(pxb_dev_list, pxb_dev);

    assert(position >= 0);



    pxb_dev_base = DEVICE(pxb_dev);

    main_host = PCI_HOST_BRIDGE(pxb_dev_base->parent_bus->parent);

    main_host_sbd = SYS_BUS_DEVICE(main_host);



    if (main_host_sbd->num_mmio > 0) {

        return g_strdup_printf(TARGET_FMT_plx ",%x",

                               main_host_sbd->mmio[0].addr, position + 1);

    }

    if (main_host_sbd->num_pio > 0) {

        return g_strdup_printf("i%04x,%x",

                               main_host_sbd->pio[0], position + 1);

    }

    return NULL;

}
