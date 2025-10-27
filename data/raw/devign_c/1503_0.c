static void mpic_irq_raise(openpic_t *mpp, int n_CPU, IRQ_src_t *src)

{

    int n_ci = IDR_CI0 - n_CPU;



    if(test_bit(&src->ide, n_ci)) {

        qemu_irq_raise(mpp->dst[n_CPU].irqs[OPENPIC_OUTPUT_CINT]);

    }

    else {

        qemu_irq_raise(mpp->dst[n_CPU].irqs[OPENPIC_OUTPUT_INT]);

    }

}
