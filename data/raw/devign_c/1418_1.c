static DeviceState *sun4c_intctl_init(target_phys_addr_t addr,

                                      qemu_irq *parent_irq)

{

    DeviceState *dev;

    SysBusDevice *s;

    unsigned int i;



    dev = qdev_create(NULL, "sun4c_intctl");

    qdev_init(dev);



    s = sysbus_from_qdev(dev);



    for (i = 0; i < MAX_PILS; i++) {

        sysbus_connect_irq(s, i, parent_irq[i]);

    }

    sysbus_mmio_map(s, 0, addr);



    return dev;

}
