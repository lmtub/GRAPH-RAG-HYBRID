static inline void menelaus_rtc_start(MenelausState *s)

{

    s->rtc.next += qemu_get_clock(rt_clock);

    qemu_mod_timer(s->rtc.hz_tm, s->rtc.next);

}
