static int nvic_pending_prio(NVICState *s)

{

    /* return the priority of the current pending interrupt,

     * or NVIC_NOEXC_PRIO if no interrupt is pending

     */

    return s->vectpending ? s->vectors[s->vectpending].prio : NVIC_NOEXC_PRIO;

}
