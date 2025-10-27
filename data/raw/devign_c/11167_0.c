static void omap_rtc_reset(struct omap_rtc_s *s)

{

    struct tm tm;



    s->interrupts = 0;

    s->comp_reg = 0;

    s->running = 0;

    s->pm_am = 0;

    s->auto_comp = 0;

    s->round = 0;

    s->tick = qemu_get_clock(rt_clock);

    memset(&s->alarm_tm, 0, sizeof(s->alarm_tm));

    s->alarm_tm.tm_mday = 0x01;

    s->status = 1 << 7;

    qemu_get_timedate(&tm, 0);

    s->ti = mktimegm(&tm);



    omap_rtc_alarm_update(s);

    omap_rtc_tick(s);

}
