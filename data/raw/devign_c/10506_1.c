static inline void menelaus_rtc_start(struct menelaus_s *s)

{

    s->rtc.next =+ qemu_get_clock(rt_clock);

    qemu_mod_timer(s->rtc.hz, s->rtc.next);

}
