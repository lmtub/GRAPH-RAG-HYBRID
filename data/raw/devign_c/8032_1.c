static int lance_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    SysBusPCNetState *d = SYSBUS_PCNET(dev);

    PCNetState *s = &d->state;



    memory_region_init_io(&s->mmio, OBJECT(d), &lance_mem_ops, d,

                          "lance-mmio", 4);



    qdev_init_gpio_in(dev, parent_lance_reset, 1);



    sysbus_init_mmio(sbd, &s->mmio);



    sysbus_init_irq(sbd, &s->irq);



    s->phys_mem_read = ledma_memory_read;

    s->phys_mem_write = ledma_memory_write;

    return pcnet_common_init(dev, s, &net_lance_info);

}
