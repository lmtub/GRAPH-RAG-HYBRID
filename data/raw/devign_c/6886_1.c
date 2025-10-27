static int lance_init(SysBusDevice *dev)

{

    SysBusPCNetState *d = FROM_SYSBUS(SysBusPCNetState, dev);

    PCNetState *s = &d->state;



    memory_region_init_io(&s->mmio, &lance_mem_ops, s, "lance-mmio", 4);



    qdev_init_gpio_in(&dev->qdev, parent_lance_reset, 1);



    sysbus_init_mmio_region(dev, &s->mmio);



    sysbus_init_irq(dev, &s->irq);



    s->phys_mem_read = ledma_memory_read;

    s->phys_mem_write = ledma_memory_write;

    return pcnet_common_init(&dev->qdev, s, &net_lance_info);

}
