void gic_update(GICState *s)

{

    int best_irq;

    int best_prio;

    int irq;

    int irq_level, fiq_level;

    int cpu;

    int cm;



    for (cpu = 0; cpu < s->num_cpu; cpu++) {

        cm = 1 << cpu;

        s->current_pending[cpu] = 1023;

        if (!(s->ctlr & (GICD_CTLR_EN_GRP0 | GICD_CTLR_EN_GRP1))

            || !(s->cpu_ctlr[cpu] & (GICC_CTLR_EN_GRP0 | GICC_CTLR_EN_GRP1))) {

            qemu_irq_lower(s->parent_irq[cpu]);

            qemu_irq_lower(s->parent_fiq[cpu]);

            continue;

        }

        best_prio = 0x100;

        best_irq = 1023;

        for (irq = 0; irq < s->num_irq; irq++) {

            if (GIC_TEST_ENABLED(irq, cm) && gic_test_pending(s, irq, cm) &&

                (irq < GIC_INTERNAL || GIC_TARGET(irq) & cm)) {

                if (GIC_GET_PRIORITY(irq, cpu) < best_prio) {

                    best_prio = GIC_GET_PRIORITY(irq, cpu);

                    best_irq = irq;

                }

            }

        }



        if (best_irq != 1023) {

            trace_gic_update_bestirq(cpu, best_irq, best_prio,

                s->priority_mask[cpu], s->running_priority[cpu]);

        }



        irq_level = fiq_level = 0;



        if (best_prio < s->priority_mask[cpu]) {

            s->current_pending[cpu] = best_irq;

            if (best_prio < s->running_priority[cpu]) {

                int group = GIC_TEST_GROUP(best_irq, cm);



                if (extract32(s->ctlr, group, 1) &&

                    extract32(s->cpu_ctlr[cpu], group, 1)) {

                    if (group == 0 && s->cpu_ctlr[cpu] & GICC_CTLR_FIQ_EN) {

                        DPRINTF("Raised pending FIQ %d (cpu %d)\n",

                                best_irq, cpu);

                        fiq_level = 1;


                    } else {

                        DPRINTF("Raised pending IRQ %d (cpu %d)\n",

                                best_irq, cpu);

                        irq_level = 1;


                    }

                }

            }

        }



        qemu_set_irq(s->parent_irq[cpu], irq_level);

        qemu_set_irq(s->parent_fiq[cpu], fiq_level);

    }

}