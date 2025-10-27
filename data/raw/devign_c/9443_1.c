static void ram_init(target_phys_addr_t addr, ram_addr_t RAM_size)

{

    DeviceState *dev;

    SysBusDevice *s;

    RamDevice *d;



    /* allocate RAM */

    dev = qdev_create(NULL, "memory");

    s = sysbus_from_qdev(dev);



    d = FROM_SYSBUS(RamDevice, s);

    d->size = RAM_size;

    qdev_init(dev);



    sysbus_mmio_map(s, 0, addr);

}
