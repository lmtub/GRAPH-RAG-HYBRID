static int tusb6010_init(SysBusDevice *dev)

{

    TUSBState *s = FROM_SYSBUS(TUSBState, dev);

    qemu_irq *musb_irqs;

    int i;

    s->otg_timer = qemu_new_timer_ns(vm_clock, tusb_otg_tick, s);

    s->pwr_timer = qemu_new_timer_ns(vm_clock, tusb_power_tick, s);

    memory_region_init_io(&s->iomem[1], &tusb_async_ops, s, "tusb-async",

                          UINT32_MAX);

    sysbus_init_mmio_region(dev, &s->iomem[0]);

    sysbus_init_mmio_region(dev, &s->iomem[1]);

    sysbus_init_irq(dev, &s->irq);

    qdev_init_gpio_in(&dev->qdev, tusb6010_irq, musb_irq_max + 1);

    musb_irqs = g_new0(qemu_irq, musb_irq_max);

    for (i = 0; i < musb_irq_max; i++) {

        musb_irqs[i] = qdev_get_gpio_in(&dev->qdev, i + 1);

    }

    s->musb = musb_init(musb_irqs);

    return 0;

}
