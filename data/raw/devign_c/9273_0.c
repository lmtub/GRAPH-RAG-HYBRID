static void openpic_irq_raise(openpic_t *opp, int n_CPU, IRQ_src_t *src)

{

    int n_ci = IDR_CI0 - n_CPU;



    if ((opp->flags & OPENPIC_FLAG_IDE_CRIT) && test_bit(&src->ide, n_ci)) {

        qemu_irq_raise(opp->dst[n_CPU].irqs[OPENPIC_OUTPUT_CINT]);

    } else {

        qemu_irq_raise(opp->dst[n_CPU].irqs[OPENPIC_OUTPUT_INT]);

    }

}
