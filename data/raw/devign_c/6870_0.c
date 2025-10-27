PCIBus *pci_pmac_init(qemu_irq *pic,

                      MemoryRegion *address_space_mem,

                      MemoryRegion *address_space_io)

{

    DeviceState *dev;

    SysBusDevice *s;

    PCIHostState *h;

    UNINState *d;



    /* Use values found on a real PowerMac */

    /* Uninorth main bus */

    dev = qdev_create(NULL, TYPE_UNI_NORTH_PCI_HOST_BRIDGE);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    h = PCI_HOST_BRIDGE(s);

    d = UNI_NORTH_PCI_HOST_BRIDGE(dev);

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



#if 0

    pci_create_simple(h->bus, PCI_DEVFN(11, 0), "uni-north");

#endif



    sysbus_mmio_map(s, 0, 0xf2800000);

    sysbus_mmio_map(s, 1, 0xf2c00000);



    /* DEC 21154 bridge */

#if 0

    /* XXX: not activated as PPC BIOS doesn't handle multiple buses properly */

    pci_create_simple(h->bus, PCI_DEVFN(12, 0), "dec-21154");

#endif



    /* Uninorth AGP bus */

    pci_create_simple(h->bus, PCI_DEVFN(11, 0), "uni-north-agp");

    dev = qdev_create(NULL, TYPE_UNI_NORTH_AGP_HOST_BRIDGE);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(s, 0, 0xf0800000);

    sysbus_mmio_map(s, 1, 0xf0c00000);



    /* Uninorth internal bus */

#if 0

    /* XXX: not needed for now */

    pci_create_simple(h->bus, PCI_DEVFN(14, 0),

                      "uni-north-internal-pci");

    dev = qdev_create(NULL, TYPE_UNI_NORTH_INTERNAL_PCI_HOST_BRIDGE);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(s, 0, 0xf4800000);

    sysbus_mmio_map(s, 1, 0xf4c00000);

#endif



    return h->bus;

}
