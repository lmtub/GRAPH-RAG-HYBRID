static void gic_complete_irq(gic_state * s, int cpu, int irq)

{

    int update = 0;

    int cm = 1 << cpu;

    DPRINTF("EOI %d\n", irq);

    if (s->running_irq[cpu] == 1023)

        return; /* No active IRQ.  */

    if (irq != 1023) {

        /* Mark level triggered interrupts as pending if they are still

           raised.  */

        if (!GIC_TEST_TRIGGER(irq) && GIC_TEST_ENABLED(irq, cm)

                && GIC_TEST_LEVEL(irq, cm) && (GIC_TARGET(irq) & cm) != 0) {

            DPRINTF("Set %d pending mask %x\n", irq, cm);

            GIC_SET_PENDING(irq, cm);

            update = 1;

        }

    }

    if (irq != s->running_irq[cpu]) {

        /* Complete an IRQ that is not currently running.  */

        int tmp = s->running_irq[cpu];

        while (s->last_active[tmp][cpu] != 1023) {

            if (s->last_active[tmp][cpu] == irq) {

                s->last_active[tmp][cpu] = s->last_active[irq][cpu];

                break;

            }

            tmp = s->last_active[tmp][cpu];

        }

        if (update) {

            gic_update(s);

        }

    } else {

        /* Complete the current running IRQ.  */

        gic_set_running_irq(s, cpu, s->last_active[s->running_irq[cpu]][cpu]);

    }

}
