static int exynos4210_combiner_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    Exynos4210CombinerState *s = EXYNOS4210_COMBINER(dev);

    unsigned int i;



    /* Allocate general purpose input signals and connect a handler to each of

     * them */

    qdev_init_gpio_in(dev, exynos4210_combiner_handler, IIC_NIRQ);



    /* Connect SysBusDev irqs to device specific irqs */

    for (i = 0; i < IIC_NIRQ; i++) {

        sysbus_init_irq(sbd, &s->output_irq[i]);

    }



    memory_region_init_io(&s->iomem, OBJECT(s), &exynos4210_combiner_ops, s,

                          "exynos4210-combiner", IIC_REGION_SIZE);

    sysbus_init_mmio(sbd, &s->iomem);



    return 0;

}
