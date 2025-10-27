uint32_t gic_acknowledge_irq(GICState *s, int cpu)

{

    int ret, irq, src;

    int cm = 1 << cpu;

    irq = s->current_pending[cpu];

    if (irq == 1023

            || GIC_GET_PRIORITY(irq, cpu) >= s->running_priority[cpu]) {

        DPRINTF("ACK no pending IRQ\n");

        return 1023;

    }

    s->last_active[irq][cpu] = s->running_irq[cpu];



    if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

        /* Clear pending flags for both level and edge triggered interrupts.

         * Level triggered IRQs will be reasserted once they become inactive.

         */

        GIC_CLEAR_PENDING(irq, GIC_TEST_MODEL(irq) ? ALL_CPU_MASK : cm);

        ret = irq;

    } else {

        if (irq < GIC_NR_SGIS) {

            /* Lookup the source CPU for the SGI and clear this in the

             * sgi_pending map.  Return the src and clear the overall pending

             * state on this CPU if the SGI is not pending from any CPUs.

             */

            assert(s->sgi_pending[irq][cpu] != 0);

            src = ctz32(s->sgi_pending[irq][cpu]);

            s->sgi_pending[irq][cpu] &= ~(1 << src);

            if (s->sgi_pending[irq][cpu] == 0) {

                GIC_CLEAR_PENDING(irq, GIC_TEST_MODEL(irq) ? ALL_CPU_MASK : cm);

            }

            ret = irq | ((src & 0x7) << 10);

        } else {

            /* Clear pending state for both level and edge triggered

             * interrupts. (level triggered interrupts with an active line

             * remain pending, see gic_test_pending)

             */

            GIC_CLEAR_PENDING(irq, GIC_TEST_MODEL(irq) ? ALL_CPU_MASK : cm);

            ret = irq;

        }

    }



    gic_set_running_irq(s, cpu, irq);

    DPRINTF("ACK %d\n", irq);

    return ret;

}
