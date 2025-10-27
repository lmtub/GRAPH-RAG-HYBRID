static void gem_init(NICInfo *nd, uint32_t base, qemu_irq irq)

{

    DeviceState *dev;

    SysBusDevice *s;



    qemu_check_nic_model(nd, "cadence_gem");

    dev = qdev_create(NULL, "cadence_gem");

    qdev_set_nic_properties(dev, nd);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(s, 0, base);

    sysbus_connect_irq(s, 0, irq);

}
