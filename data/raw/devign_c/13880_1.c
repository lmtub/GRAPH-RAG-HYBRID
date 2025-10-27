static void hpet_set_timer(HPETTimer *t)

{

    uint64_t diff;

    uint32_t wrap_diff;  /* how many ticks until we wrap? */

    uint64_t cur_tick = hpet_get_ticks();



    /* whenever new timer is being set up, make sure wrap_flag is 0 */

    t->wrap_flag = 0;

    diff = hpet_calculate_diff(t, cur_tick);



    /* hpet spec says in one-shot 32-bit mode, generate an interrupt when

     * counter wraps in addition to an interrupt with comparator match.

     */

    if (t->config & HPET_TN_32BIT && !timer_is_periodic(t)) {

        wrap_diff = 0xffffffff - (uint32_t)cur_tick;

        if (wrap_diff < (uint32_t)diff) {

            diff = wrap_diff;

            t->wrap_flag = 1;

        }

    }

    qemu_mod_timer(t->qemu_timer, qemu_get_clock(vm_clock)

                   + (int64_t)ticks_to_ns(diff));

}
