static void openpic_set_irq(void *opaque, int n_IRQ, int level)

{

    OpenPICState *opp = opaque;

    IRQSource *src;



    if (n_IRQ >= MAX_IRQ) {

        fprintf(stderr, "%s: IRQ %d out of range\n", __func__, n_IRQ);

        abort();

    }



    src = &opp->src[n_IRQ];

    DPRINTF("openpic: set irq %d = %d ivpr=0x%08x\n",

            n_IRQ, level, src->ivpr);

    if (src->level) {

        /* level-sensitive irq */

        src->pending = level;

        if (!level) {

            src->ivpr &= ~IVPR_ACTIVITY_MASK;

        }

    } else {

        /* edge-sensitive irq */

        if (level) {

            src->pending = 1;

        }

    }

    openpic_update_irq(opp, n_IRQ);

}
