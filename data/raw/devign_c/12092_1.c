static int omap_i2c_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    OMAPI2CState *s = OMAP_I2C(dev);



    if (!s->fclk) {

        hw_error("omap_i2c: fclk not connected\n");

    }

    if (s->revision >= OMAP2_INTR_REV && !s->iclk) {

        /* Note that OMAP1 doesn't have a separate interface clock */

        hw_error("omap_i2c: iclk not connected\n");

    }

    sysbus_init_irq(sbd, &s->irq);

    sysbus_init_irq(sbd, &s->drq[0]);

    sysbus_init_irq(sbd, &s->drq[1]);

    memory_region_init_io(&s->iomem, OBJECT(s), &omap_i2c_ops, s, "omap.i2c",

                          (s->revision < OMAP2_INTR_REV) ? 0x800 : 0x1000);

    sysbus_init_mmio(sbd, &s->iomem);

    s->bus = i2c_init_bus(dev, NULL);

    return 0;

}
