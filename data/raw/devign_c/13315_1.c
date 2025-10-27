m48t59_t *m48t59_init (qemu_irq IRQ, target_phys_addr_t mem_base,

                       uint32_t io_base, uint16_t size,

                       int type)

{

    DeviceState *dev;

    SysBusDevice *s;

    M48t59SysBusState *d;



    dev = qdev_create(NULL, "m48t59");

    qdev_prop_set_uint32(dev, "type", type);

    qdev_prop_set_uint32(dev, "size", size);

    qdev_prop_set_uint32(dev, "io_base", io_base);

    qdev_init(dev);

    s = sysbus_from_qdev(dev);

    sysbus_connect_irq(s, 0, IRQ);

    if (io_base != 0) {

        register_ioport_read(io_base, 0x04, 1, NVRAM_readb, s);

        register_ioport_write(io_base, 0x04, 1, NVRAM_writeb, s);

    }

    if (mem_base != 0) {

        sysbus_mmio_map(s, 0, mem_base);

    }



    d = FROM_SYSBUS(M48t59SysBusState, s);



    return &d->state;

}
