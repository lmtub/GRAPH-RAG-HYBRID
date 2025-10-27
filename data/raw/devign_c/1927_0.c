static void armv7m_nvic_realize(DeviceState *dev, Error **errp)

{

    NVICState *s = NVIC(dev);

    SysBusDevice *systick_sbd;

    Error *err = NULL;



    s->cpu = ARM_CPU(qemu_get_cpu(0));

    assert(s->cpu);



    if (s->num_irq > NVIC_MAX_IRQ) {

        error_setg(errp, "num-irq %d exceeds NVIC maximum", s->num_irq);

        return;

    }



    qdev_init_gpio_in(dev, set_irq_level, s->num_irq);



    /* include space for internal exception vectors */

    s->num_irq += NVIC_FIRST_IRQ;



    object_property_set_bool(OBJECT(&s->systick), true, "realized", &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }

    systick_sbd = SYS_BUS_DEVICE(&s->systick);

    sysbus_connect_irq(systick_sbd, 0,

                       qdev_get_gpio_in_named(dev, "systick-trigger", 0));



    /* The NVIC and System Control Space (SCS) starts at 0xe000e000

     * and looks like this:

     *  0x004 - ICTR

     *  0x010 - 0xff - systick

     *  0x100..0x7ec - NVIC

     *  0x7f0..0xcff - Reserved

     *  0xd00..0xd3c - SCS registers

     *  0xd40..0xeff - Reserved or Not implemented

     *  0xf00 - STIR

     */

    memory_region_init(&s->container, OBJECT(s), "nvic", 0x1000);

    /* The system register region goes at the bottom of the priority

     * stack as it covers the whole page.

     */

    memory_region_init_io(&s->sysregmem, OBJECT(s), &nvic_sysreg_ops, s,

                          "nvic_sysregs", 0x1000);

    memory_region_add_subregion(&s->container, 0, &s->sysregmem);

    memory_region_add_subregion_overlap(&s->container, 0x10,

                                        sysbus_mmio_get_region(systick_sbd, 0),

                                        1);



    sysbus_init_mmio(SYS_BUS_DEVICE(dev), &s->container);

}
