void gic_complete_irq(GICState *s, int cpu, int irq)

{

    int update = 0;

    int cm = 1 << cpu;

    DPRINTF("EOI %d\n", irq);

    if (irq >= s->num_irq) {

        /* This handles two cases:

         * 1. If software writes the ID of a spurious interrupt [ie 1023]

         * to the GICC_EOIR, the GIC ignores that write.

         * 2. If software writes the number of a non-existent interrupt

         * this must be a subcase of "value written does not match the last

         * valid interrupt value read from the Interrupt Acknowledge

         * register" and so this is UNPREDICTABLE. We choose to ignore it.

         */

        return;

    }

    if (s->running_irq[cpu] == 1023)

        return; /* No active IRQ.  */



    if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

        /* Mark level triggered interrupts as pending if they are still

           raised.  */

        if (!GIC_TEST_EDGE_TRIGGER(irq) && GIC_TEST_ENABLED(irq, cm)

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
