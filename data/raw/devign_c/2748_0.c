static inline void omap_gp_timer_trigger(struct omap_gp_timer_s *timer)

{

    if (timer->pt)

        /* TODO in overflow-and-match mode if the first event to

         * occurs is the match, don't toggle.  */

        omap_gp_timer_out(timer, !timer->out_val);

    else

        /* TODO inverted pulse on timer->out_val == 1?  */

        qemu_irq_pulse(timer->out);

}
