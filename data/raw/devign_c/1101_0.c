static void imx_epit_reload_compare_timer(IMXEPITState *s)

{

    if ((s->cr & (CR_EN | CR_OCIEN)) == (CR_EN | CR_OCIEN))  {

        /* if the compare feature is on and timers are running */

        uint32_t tmp = imx_epit_update_count(s);

        uint64_t next;

        if (tmp > s->cmp) {

            /* It'll fire in this round of the timer */

            next = tmp - s->cmp;

        } else { /* catch it next time around */

            next = tmp - s->cmp + ((s->cr & CR_RLD) ? TIMER_MAX : s->lr);

        }

        ptimer_set_count(s->timer_cmp, next);

    }

}
