static void virtio_pci_realize(PCIDevice *pci_dev, Error **errp)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(pci_dev);

    VirtioPCIClass *k = VIRTIO_PCI_GET_CLASS(pci_dev);



    /*

     * virtio pci bar layout used by default.

     * subclasses can re-arrange things if needed.

     *

     *   region 0   --  virtio legacy io bar

     *   region 1   --  msi-x bar

     *   region 4+5 --  virtio modern memory (64bit) bar

     *

     */

    proxy->legacy_io_bar  = 0;

    proxy->msix_bar       = 1;

    proxy->modern_io_bar  = 2;

    proxy->modern_mem_bar = 4;



    proxy->common.offset = 0x0;

    proxy->common.size = 0x1000;

    proxy->common.type = VIRTIO_PCI_CAP_COMMON_CFG;



    proxy->isr.offset = 0x1000;

    proxy->isr.size = 0x1000;

    proxy->isr.type = VIRTIO_PCI_CAP_ISR_CFG;



    proxy->device.offset = 0x2000;

    proxy->device.size = 0x1000;

    proxy->device.type = VIRTIO_PCI_CAP_DEVICE_CFG;



    proxy->notify.offset = 0x3000;

    proxy->notify.size =

        QEMU_VIRTIO_PCI_QUEUE_MEM_MULT * VIRTIO_QUEUE_MAX;

    proxy->notify.type = VIRTIO_PCI_CAP_NOTIFY_CFG;



    proxy->notify_pio.offset = 0x0;

    proxy->notify_pio.size = 0x4;

    proxy->notify_pio.type = VIRTIO_PCI_CAP_NOTIFY_CFG;



    /* subclasses can enforce modern, so do this unconditionally */

    memory_region_init(&proxy->modern_bar, OBJECT(proxy), "virtio-pci",

                       2 * QEMU_VIRTIO_PCI_QUEUE_MEM_MULT *

                       VIRTIO_QUEUE_MAX);



    memory_region_init_alias(&proxy->modern_cfg,

                             OBJECT(proxy),

                             "virtio-pci-cfg",

                             &proxy->modern_bar,

                             0,

                             memory_region_size(&proxy->modern_bar));



    address_space_init(&proxy->modern_as, &proxy->modern_cfg, "virtio-pci-cfg-as");



    if (!(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_PCIE)

        && !(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_MODERN)

        && pci_bus_is_express(pci_dev->bus)

        && !pci_bus_is_root(pci_dev->bus)) {

        int pos;



        pci_dev->cap_present |= QEMU_PCI_CAP_EXPRESS;

        pos = pcie_endpoint_cap_init(pci_dev, 0);

        assert(pos > 0);



        pos = pci_add_capability(pci_dev, PCI_CAP_ID_PM, 0, PCI_PM_SIZEOF);

        assert(pos > 0);



        /*

         * Indicates that this function complies with revision 1.2 of the

         * PCI Power Management Interface Specification.

         */

        pci_set_word(pci_dev->config + pos + PCI_PM_PMC, 0x3);

    }



    virtio_pci_bus_new(&proxy->bus, sizeof(proxy->bus), proxy);

    if (k->realize) {

        k->realize(proxy, errp);

    }

}
