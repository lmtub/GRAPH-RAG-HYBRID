static void gic_set_irq(void *opaque, int irq, int level)

{

    /* Meaning of the 'irq' parameter:

     *  [0..N-1] : external interrupts

     *  [N..N+31] : PPI (internal) interrupts for CPU 0

     *  [N+32..N+63] : PPI (internal interrupts for CPU 1

     *  ...

     */

    GICState *s = (GICState *)opaque;

    int cm, target;

    if (irq < (s->num_irq - GIC_INTERNAL)) {

        /* The first external input line is internal interrupt 32.  */

        cm = ALL_CPU_MASK;

        irq += GIC_INTERNAL;

        target = GIC_TARGET(irq);

    } else {

        int cpu;

        irq -= (s->num_irq - GIC_INTERNAL);

        cpu = irq / GIC_INTERNAL;

        irq %= GIC_INTERNAL;

        cm = 1 << cpu;

        target = cm;

    }



    assert(irq >= GIC_NR_SGIS);



    if (level == GIC_TEST_LEVEL(irq, cm)) {

        return;

    }



    if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

        gic_set_irq_11mpcore(s, irq, level, cm, target);

    } else {

        gic_set_irq_generic(s, irq, level, cm, target);

    }




    gic_update(s);

}