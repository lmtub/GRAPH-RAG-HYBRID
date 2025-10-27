static void pxa2xx_rtc_piupdate(PXA2xxRTCState *s)

{

    int64_t rt = qemu_get_clock(rt_clock);

    if (s->rtsr & (1 << 15))

        s->last_swcr += rt - s->last_pi;

    s->last_pi = rt;

}
