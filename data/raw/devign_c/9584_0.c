MemoryRegion *escc_init(target_phys_addr_t base, qemu_irq irqA, qemu_irq irqB,

              CharDriverState *chrA, CharDriverState *chrB,

              int clock, int it_shift)

{

    DeviceState *dev;

    SysBusDevice *s;

    SerialState *d;



    dev = qdev_create(NULL, "escc");

    qdev_prop_set_uint32(dev, "disabled", 0);

    qdev_prop_set_uint32(dev, "frequency", clock);

    qdev_prop_set_uint32(dev, "it_shift", it_shift);

    qdev_prop_set_chr(dev, "chrB", chrB);

    qdev_prop_set_chr(dev, "chrA", chrA);

    qdev_prop_set_uint32(dev, "chnBtype", ser);

    qdev_prop_set_uint32(dev, "chnAtype", ser);

    qdev_init_nofail(dev);

    s = sysbus_from_qdev(dev);

    sysbus_connect_irq(s, 0, irqB);

    sysbus_connect_irq(s, 1, irqA);

    if (base) {

        sysbus_mmio_map(s, 0, base);

    }



    d = FROM_SYSBUS(SerialState, s);

    return &d->mmio;

}
