static void arm_gic_common_reset(DeviceState *dev)

{

    GICState *s = ARM_GIC_COMMON(dev);

    int i;

    memset(s->irq_state, 0, GIC_MAXIRQ * sizeof(gic_irq_state));

    for (i = 0 ; i < s->num_cpu; i++) {

        if (s->revision == REV_11MPCORE) {

            s->priority_mask[i] = 0xf0;

        } else {

            s->priority_mask[i] = 0;

        }

        s->current_pending[i] = 1023;

        s->running_irq[i] = 1023;

        s->running_priority[i] = 0x100;

        s->cpu_enabled[i] = false;

    }

    for (i = 0; i < GIC_NR_SGIS; i++) {

        GIC_SET_ENABLED(i, ALL_CPU_MASK);

        GIC_SET_EDGE_TRIGGER(i);

    }

    if (s->num_cpu == 1) {

        /* For uniprocessor GICs all interrupts always target the sole CPU */

        for (i = 0; i < GIC_MAXIRQ; i++) {

            s->irq_target[i] = 1;

        }

    }

    s->ctlr = 0;

}
