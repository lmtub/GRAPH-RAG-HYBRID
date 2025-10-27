static void omap_mcbsp_sink_tick(void *opaque)

{

    struct omap_mcbsp_s *s = (struct omap_mcbsp_s *) opaque;

    static const int bps[8] = { 0, 1, 1, 2, 2, 2, -255, -255 };



    if (!s->tx_rate)

        return;

    if (s->tx_req)

        printf("%s: Tx FIFO underrun\n", __FUNCTION__);



    s->tx_req = s->tx_rate << bps[(s->xcr[0] >> 5) & 7];



    omap_mcbsp_tx_newdata(s);

    timer_mod(s->sink_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +

                   NANOSECONDS_PER_SECOND);

}
