void slavio_serial_ms_kbd_init(target_phys_addr_t base, qemu_irq irq,

                               int disabled, int clock, int it_shift)

{

    DeviceState *dev;

    SysBusDevice *s;



    dev = qdev_create(NULL, "escc");

    qdev_prop_set_uint32(dev, "disabled", disabled);

    qdev_prop_set_uint32(dev, "frequency", clock);

    qdev_prop_set_uint32(dev, "it_shift", it_shift);

    qdev_prop_set_chr(dev, "chrB", NULL);

    qdev_prop_set_chr(dev, "chrA", NULL);

    qdev_prop_set_uint32(dev, "chnBtype", mouse);

    qdev_prop_set_uint32(dev, "chnAtype", kbd);

    qdev_init(dev);

    s = sysbus_from_qdev(dev);

    sysbus_connect_irq(s, 0, irq);

    sysbus_connect_irq(s, 1, irq);

    sysbus_mmio_map(s, 0, base);

}
