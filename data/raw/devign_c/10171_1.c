static void openpic_irq_raise(OpenPICState *opp, int n_CPU, IRQ_src_t *src)

{

    int n_ci = IDR_CI0_SHIFT - n_CPU;



    if ((opp->flags & OPENPIC_FLAG_IDE_CRIT) && (src->ide & (1 << n_ci))) {

        qemu_irq_raise(opp->dst[n_CPU].irqs[OPENPIC_OUTPUT_CINT]);

    } else {

        qemu_irq_raise(opp->dst[n_CPU].irqs[OPENPIC_OUTPUT_INT]);

    }

}
