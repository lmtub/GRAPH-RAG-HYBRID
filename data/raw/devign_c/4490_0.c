static int pxa2xx_rtc_init(SysBusDevice *dev)

{

    PXA2xxRTCState *s = FROM_SYSBUS(PXA2xxRTCState, dev);

    struct tm tm;

    int wom;

    int iomemtype;



    s->rttr = 0x7fff;

    s->rtsr = 0;



    qemu_get_timedate(&tm, 0);

    wom = ((tm.tm_mday - 1) / 7) + 1;



    s->last_rcnr = (uint32_t) mktimegm(&tm);

    s->last_rdcr = (wom << 20) | ((tm.tm_wday + 1) << 17) |

            (tm.tm_hour << 12) | (tm.tm_min << 6) | tm.tm_sec;

    s->last_rycr = ((tm.tm_year + 1900) << 9) |

            ((tm.tm_mon + 1) << 5) | tm.tm_mday;

    s->last_swcr = (tm.tm_hour << 19) |

            (tm.tm_min << 13) | (tm.tm_sec << 7);

    s->last_rtcpicr = 0;

    s->last_hz = s->last_sw = s->last_pi = qemu_get_clock(rt_clock);



    s->rtc_hz    = qemu_new_timer(rt_clock, pxa2xx_rtc_hz_tick,    s);

    s->rtc_rdal1 = qemu_new_timer(rt_clock, pxa2xx_rtc_rdal1_tick, s);

    s->rtc_rdal2 = qemu_new_timer(rt_clock, pxa2xx_rtc_rdal2_tick, s);

    s->rtc_swal1 = qemu_new_timer(rt_clock, pxa2xx_rtc_swal1_tick, s);

    s->rtc_swal2 = qemu_new_timer(rt_clock, pxa2xx_rtc_swal2_tick, s);

    s->rtc_pi    = qemu_new_timer(rt_clock, pxa2xx_rtc_pi_tick,    s);



    sysbus_init_irq(dev, &s->rtc_irq);



    iomemtype = cpu_register_io_memory(pxa2xx_rtc_readfn,

                    pxa2xx_rtc_writefn, s, DEVICE_NATIVE_ENDIAN);

    sysbus_init_mmio(dev, 0x10000, iomemtype);



    return 0;

}
