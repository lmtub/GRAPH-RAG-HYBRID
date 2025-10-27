static int rtc_initfn(ISADevice *dev)

{

    RTCState *s = DO_UPCAST(RTCState, dev, dev);

    int base = 0x70;

    int isairq = 8;



    isa_init_irq(dev, &s->irq, isairq);



    s->cmos_data[RTC_REG_A] = 0x26;

    s->cmos_data[RTC_REG_B] = 0x02;

    s->cmos_data[RTC_REG_C] = 0x00;

    s->cmos_data[RTC_REG_D] = 0x80;



    rtc_set_date_from_host(s);



    s->periodic_timer = qemu_new_timer(rtc_clock, rtc_periodic_timer, s);

#ifdef TARGET_I386

    if (rtc_td_hack)

        s->coalesced_timer =

            qemu_new_timer(rtc_clock, rtc_coalesced_timer, s);

#endif

    s->second_timer = qemu_new_timer(rtc_clock, rtc_update_second, s);

    s->second_timer2 = qemu_new_timer(rtc_clock, rtc_update_second2, s);



    s->next_second_time =

        qemu_get_clock(rtc_clock) + (get_ticks_per_sec() * 99) / 100;

    qemu_mod_timer(s->second_timer2, s->next_second_time);



    register_ioport_write(base, 2, 1, cmos_ioport_write, s);

    register_ioport_read(base, 2, 1, cmos_ioport_read, s);



    register_savevm("mc146818rtc", base, 1, rtc_save, rtc_load, s);

#ifdef TARGET_I386

    if (rtc_td_hack)

        register_savevm("mc146818rtc-td", base, 1, rtc_save_td, rtc_load_td, s);

#endif

    qemu_register_reset(rtc_reset, s);

    return 0;

}
