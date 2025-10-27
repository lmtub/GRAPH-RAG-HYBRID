static void kvm_arm_gic_put(GICState *s)

{

    uint32_t reg;

    int i;

    int cpu;

    int num_cpu;

    int num_irq;



    if (!kvm_arm_gic_can_save_restore(s)) {

            DPRINTF("Cannot put kernel gic state, no kernel interface");

            return;

    }



    /* Note: We do the restore in a slightly different order than the save

     * (where the order doesn't matter and is simply ordered according to the

     * register offset values */



    /*****************************************************************

     * Distributor State

     */



    /* s->enabled -> GICD_CTLR */

    reg = s->enabled;

    kvm_gicd_access(s, 0x0, 0, &reg, true);



    /* Sanity checking on GICD_TYPER and s->num_irq, s->num_cpu */

    kvm_gicd_access(s, 0x4, 0, &reg, false);

    num_irq = ((reg & 0x1f) + 1) * 32;

    num_cpu = ((reg & 0xe0) >> 5) + 1;



    if (num_irq < s->num_irq) {

            fprintf(stderr, "Restoring %u IRQs, but kernel supports max %d\n",

                    s->num_irq, num_irq);

            abort();

    } else if (num_cpu != s->num_cpu) {

            fprintf(stderr, "Restoring %u CPU interfaces, kernel only has %d\n",

                    s->num_cpu, num_cpu);

            /* Did we not create the VCPUs in the kernel yet? */

            abort();

    }



    /* TODO: Consider checking compatibility with the IIDR ? */



    /* irq_state[n].enabled -> GICD_ISENABLERn */

    kvm_dist_put(s, 0x180, 1, s->num_irq, translate_clear);

    kvm_dist_put(s, 0x100, 1, s->num_irq, translate_enabled);



    /* irq_state[n].group -> GICD_IGROUPRn */

    kvm_dist_put(s, 0x80, 1, s->num_irq, translate_group);



    /* s->irq_target[irq] -> GICD_ITARGETSRn

     * (restore targets before pending to ensure the pending state is set on

     * the appropriate CPU interfaces in the kernel) */

    kvm_dist_put(s, 0x800, 8, s->num_irq, translate_targets);



    /* irq_state[n].trigger -> GICD_ICFGRn

     * (restore configuration registers before pending IRQs so we treat

     * level/edge correctly) */

    kvm_dist_put(s, 0xc00, 2, s->num_irq, translate_trigger);



    /* irq_state[n].pending + irq_state[n].level -> GICD_ISPENDRn */

    kvm_dist_put(s, 0x280, 1, s->num_irq, translate_clear);

    kvm_dist_put(s, 0x200, 1, s->num_irq, translate_pending);



    /* irq_state[n].active -> GICD_ISACTIVERn */

    kvm_dist_put(s, 0x380, 1, s->num_irq, translate_clear);

    kvm_dist_put(s, 0x300, 1, s->num_irq, translate_active);





    /* s->priorityX[irq] -> ICD_IPRIORITYRn */

    kvm_dist_put(s, 0x400, 8, s->num_irq, translate_priority);



    /* s->sgi_pending -> ICD_CPENDSGIRn */

    kvm_dist_put(s, 0xf10, 8, GIC_NR_SGIS, translate_clear);

    kvm_dist_put(s, 0xf20, 8, GIC_NR_SGIS, translate_sgisource);





    /*****************************************************************

     * CPU Interface(s) State

     */



    for (cpu = 0; cpu < s->num_cpu; cpu++) {

        /* s->cpu_enabled[cpu] -> GICC_CTLR */

        reg = s->cpu_enabled[cpu];

        kvm_gicc_access(s, 0x00, cpu, &reg, true);



        /* s->priority_mask[cpu] -> GICC_PMR */

        reg = (s->priority_mask[cpu] & 0xff);

        kvm_gicc_access(s, 0x04, cpu, &reg, true);



        /* s->bpr[cpu] -> GICC_BPR */

        reg = (s->bpr[cpu] & 0x7);

        kvm_gicc_access(s, 0x08, cpu, &reg, true);



        /* s->abpr[cpu] -> GICC_ABPR */

        reg = (s->abpr[cpu] & 0x7);

        kvm_gicc_access(s, 0x1c, cpu, &reg, true);



        /* s->apr[n][cpu] -> GICC_APRn */

        for (i = 0; i < 4; i++) {

            reg = s->apr[i][cpu];

            kvm_gicc_access(s, 0xd0 + i * 4, cpu, &reg, true);

        }

    }

}
