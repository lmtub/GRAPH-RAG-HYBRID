static void i82378_init(DeviceState *dev, I82378State *s)

{

    ISABus *isabus = DO_UPCAST(ISABus, qbus, qdev_get_child_bus(dev, "isa.0"));

    ISADevice *pit;

    qemu_irq *out0_irq;



    /* This device has:

       2 82C59 (irq)

       1 82C54 (pit)

       2 82C37 (dma)

       NMI

       Utility Bus Support Registers



       All devices accept byte access only, except timer

     */



    qdev_init_gpio_out(dev, s->out, 2);

    qdev_init_gpio_in(dev, i82378_request_pic_irq, 16);



    /* Workaround the fact that i8259 is not qdev'ified... */

    out0_irq = qemu_allocate_irqs(i82378_request_out0_irq, s, 1);



    /* 2 82C59 (irq) */

    s->i8259 = i8259_init(isabus, *out0_irq);

    isa_bus_irqs(isabus, s->i8259);



    /* 1 82C54 (pit) */

    pit = pit_init(isabus, 0x40, 0, NULL);



    /* speaker */

    pcspk_init(isabus, pit);



    /* 2 82C37 (dma) */

    DMA_init(1, &s->out[1]);

    isa_create_simple(isabus, "i82374");



    /* timer */

    isa_create_simple(isabus, "mc146818rtc");

}
