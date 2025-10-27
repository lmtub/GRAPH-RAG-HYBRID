void qemu_chr_generic_open(CharDriverState *s)

{

    if (s->open_timer == NULL) {

        s->open_timer = qemu_new_timer_ms(rt_clock,

                                          qemu_chr_fire_open_event, s);

        qemu_mod_timer(s->open_timer, qemu_get_clock_ms(rt_clock) - 1);

    }

}
