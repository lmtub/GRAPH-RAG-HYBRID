PCIBus *pci_grackle_init(uint32_t base, qemu_irq *pic)

{

    DeviceState *dev;

    SysBusDevice *s;

    GrackleState *d;



    dev = qdev_create(NULL, "grackle");

    qdev_init(dev);

    s = sysbus_from_qdev(dev);

    d = FROM_SYSBUS(GrackleState, s);

    d->host_state.bus = pci_register_bus(&d->busdev.qdev, "pci",

                                         pci_grackle_set_irq,

                                         pci_grackle_map_irq,

                                         pic, 0, 4);



    pci_create_simple(d->host_state.bus, 0, "grackle");



    sysbus_mmio_map(s, 0, base);

    sysbus_mmio_map(s, 1, base + 0x00200000);



    return d->host_state.bus;

}
