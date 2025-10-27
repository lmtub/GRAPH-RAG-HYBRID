void exynos4210_init_board_irqs(Exynos4210Irq *s)

{

    uint32_t grp, bit, irq_id, n;



    for (n = 0; n < EXYNOS4210_MAX_EXT_COMBINER_IN_IRQ; n++) {

        s->board_irqs[n] = qemu_irq_split(s->int_combiner_irq[n],

                s->ext_combiner_irq[n]);



        irq_id = 0;

        if (n == EXYNOS4210_COMBINER_GET_IRQ_NUM(1, 4) ||

                n == EXYNOS4210_COMBINER_GET_IRQ_NUM(12, 4)) {

            /* MCT_G0 is passed to External GIC */

            irq_id = EXT_GIC_ID_MCT_G0;

        }

        if (n == EXYNOS4210_COMBINER_GET_IRQ_NUM(1, 5) ||

                n == EXYNOS4210_COMBINER_GET_IRQ_NUM(12, 5)) {

            /* MCT_G1 is passed to External and GIC */

            irq_id = EXT_GIC_ID_MCT_G1;

        }

        if (irq_id) {

            s->board_irqs[n] = qemu_irq_split(s->int_combiner_irq[n],

                    s->ext_gic_irq[irq_id-32]);

        }



    }

    for (; n < EXYNOS4210_MAX_INT_COMBINER_IN_IRQ; n++) {

        /* these IDs are passed to Internal Combiner and External GIC */

        grp = EXYNOS4210_COMBINER_GET_GRP_NUM(n);

        bit = EXYNOS4210_COMBINER_GET_BIT_NUM(n);

        irq_id = combiner_grp_to_gic_id[grp -

                     EXYNOS4210_MAX_EXT_COMBINER_OUT_IRQ][bit];



        if (irq_id) {

            s->board_irqs[n] = qemu_irq_split(s->int_combiner_irq[n],

                    s->ext_gic_irq[irq_id-32]);

        }

    }

}
