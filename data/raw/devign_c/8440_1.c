int pic_read_irq(DeviceState *d)

{

    PICCommonState *s = DO_UPCAST(PICCommonState, dev.qdev, d);

    int irq, irq2, intno;



    irq = pic_get_irq(s);

    if (irq >= 0) {

        if (irq == 2) {

            irq2 = pic_get_irq(slave_pic);

            if (irq2 >= 0) {

                pic_intack(slave_pic, irq2);

            } else {

                /* spurious IRQ on slave controller */















                irq2 = 7;


            intno = slave_pic->irq_base + irq2;

        } else {

            intno = s->irq_base + irq;


        pic_intack(s, irq);

    } else {

        /* spurious IRQ on host controller */




        irq = 7;

        intno = s->irq_base + irq;




#if defined(DEBUG_PIC) || defined(DEBUG_IRQ_LATENCY)

    if (irq == 2) {

        irq = irq2 + 8;


#endif

#ifdef DEBUG_IRQ_LATENCY

    printf("IRQ%d latency=%0.3fus\n",

           irq,

           (double)(qemu_get_clock_ns(vm_clock) -

                    irq_time[irq]) * 1000000.0 / get_ticks_per_sec());

#endif

    DPRINTF("pic_interrupt: irq=%d\n", irq);

    return intno;
