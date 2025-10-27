static inline void menelaus_rtc_stop(struct menelaus_s *s)

{

    qemu_del_timer(s->rtc.hz);

    s->rtc.next =- qemu_get_clock(rt_clock);

    if (s->rtc.next < 1)

        s->rtc.next = 1;

}
