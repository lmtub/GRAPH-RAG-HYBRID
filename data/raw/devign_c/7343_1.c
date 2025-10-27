static void IRQ_check(OpenPICState *opp, IRQ_queue_t *q)

{

    int next, i;

    int priority;



    next = -1;

    priority = -1;



    if (!q->pending) {

        /* IRQ bitmap is empty */

        goto out;

    }



    for (i = 0; i < opp->max_irq; i++) {

        if (IRQ_testbit(q, i)) {

            DPRINTF("IRQ_check: irq %d set ipvp_pr=%d pr=%d\n",

                    i, IPVP_PRIORITY(opp->src[i].ipvp), priority);

            if (IPVP_PRIORITY(opp->src[i].ipvp) > priority) {

                next = i;

                priority = IPVP_PRIORITY(opp->src[i].ipvp);

            }

        }

    }



out:

    q->next = next;

    q->priority = priority;

}
