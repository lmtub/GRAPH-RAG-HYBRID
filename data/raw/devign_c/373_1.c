static int omap_gpio_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    struct omap_gpif_s *s = OMAP1_GPIO(dev);



    if (!s->clk) {

        hw_error("omap-gpio: clk not connected\n");

    }

    qdev_init_gpio_in(dev, omap_gpio_set, 16);

    qdev_init_gpio_out(dev, s->omap1.handler, 16);

    sysbus_init_irq(sbd, &s->omap1.irq);

    memory_region_init_io(&s->iomem, OBJECT(s), &omap_gpio_ops, &s->omap1,

                          "omap.gpio", 0x1000);

    sysbus_init_mmio(sbd, &s->iomem);

    return 0;

}
