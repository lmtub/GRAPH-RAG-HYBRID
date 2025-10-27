PCIBus *pci_pmac_u3_init(qemu_irq *pic,

                         MemoryRegion *address_space_mem,

                         MemoryRegion *address_space_io)

{

    DeviceState *dev;

    SysBusDevice *s;

    PCIHostState *h;

    UNINState *d;



    /* Uninorth AGP bus */



    dev = qdev_create(NULL, TYPE_U3_AGP_HOST_BRIDGE);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    h = PCI_HOST_BRIDGE(dev);

    d = U3_AGP_HOST_BRIDGE(dev);



    memory_region_init(&d->pci_mmio, OBJECT(d), "pci-mmio", 0x100000000ULL);

    memory_region_init_alias(&d->pci_hole, OBJECT(d), "pci-hole", &d->pci_mmio,

                             0x80000000ULL, 0x70000000ULL);

    memory_region_add_subregion(address_space_mem, 0x80000000ULL,

                                &d->pci_hole);



    h->bus = pci_register_bus(dev, "pci",

                              pci_unin_set_irq, pci_unin_map_irq,

                              pic,

                              &d->pci_mmio,

                              address_space_io,

                              PCI_DEVFN(11, 0), 4, TYPE_PCI_BUS);



    sysbus_mmio_map(s, 0, 0xf0800000);

    sysbus_mmio_map(s, 1, 0xf0c00000);



    pci_create_simple(h->bus, 11 << 3, "u3-agp");



    return h->bus;

}
