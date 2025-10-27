static inline void menelaus_rtc_stop(MenelausState *s)

{

    qemu_del_timer(s->rtc.hz_tm);

    s->rtc.next -= qemu_get_clock(rt_clock);

    if (s->rtc.next < 1)

        s->rtc.next = 1;

}
