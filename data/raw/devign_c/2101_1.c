static void IRQ_local_pipe(OpenPICState *opp, int n_CPU, int n_IRQ)

{

    IRQ_dst_t *dst;

    IRQ_src_t *src;

    int priority;



    dst = &opp->dst[n_CPU];

    src = &opp->src[n_IRQ];

    priority = IPVP_PRIORITY(src->ipvp);

    if (priority <= dst->pctp) {

        /* Too low priority */

        DPRINTF("%s: IRQ %d has too low priority on CPU %d\n",

                __func__, n_IRQ, n_CPU);

        return;

    }

    if (IRQ_testbit(&dst->raised, n_IRQ)) {

        /* Interrupt miss */

        DPRINTF("%s: IRQ %d was missed on CPU %d\n",

                __func__, n_IRQ, n_CPU);

        return;

    }

    src->ipvp |= IPVP_ACTIVITY_MASK;

    IRQ_setbit(&dst->raised, n_IRQ);

    if (priority < dst->raised.priority) {

        /* An higher priority IRQ is already raised */

        DPRINTF("%s: IRQ %d is hidden by raised IRQ %d on CPU %d\n",

                __func__, n_IRQ, dst->raised.next, n_CPU);

        return;

    }

    IRQ_get_next(opp, &dst->raised);

    if (IRQ_get_next(opp, &dst->servicing) != -1 &&

        priority <= dst->servicing.priority) {

        DPRINTF("%s: IRQ %d is hidden by servicing IRQ %d on CPU %d\n",

                __func__, n_IRQ, dst->servicing.next, n_CPU);

        /* Already servicing a higher priority IRQ */

        return;

    }

    DPRINTF("Raise OpenPIC INT output cpu %d irq %d\n", n_CPU, n_IRQ);

    openpic_irq_raise(opp, n_CPU, src);

}
