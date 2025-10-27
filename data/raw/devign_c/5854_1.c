static void ppcuic_set_irq (void *opaque, int irq_num, int level)

{

    ppcuic_t *uic;

    uint32_t mask, sr;



    uic = opaque;

    mask = 1 << (31-irq_num);

    LOG_UIC("%s: irq %d level %d uicsr %08" PRIx32

                " mask %08" PRIx32 " => %08" PRIx32 " %08" PRIx32 "\n",

                __func__, irq_num, level,

                uic->uicsr, mask, uic->uicsr & mask, level << irq_num);

    if (irq_num < 0 || irq_num > 31)

        return;

    sr = uic->uicsr;



    /* Update status register */

    if (uic->uictr & mask) {

        /* Edge sensitive interrupt */

        if (level == 1)

            uic->uicsr |= mask;

    } else {

        /* Level sensitive interrupt */

        if (level == 1) {

            uic->uicsr |= mask;

            uic->level |= mask;

        } else {

            uic->uicsr &= ~mask;

            uic->level &= ~mask;

        }

    }

    LOG_UIC("%s: irq %d level %d sr %" PRIx32 " => "

                "%08" PRIx32 "\n", __func__, irq_num, level, uic->uicsr, sr);

    if (sr != uic->uicsr)

        ppcuic_trigger_irq(uic);

}
