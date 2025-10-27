static void rtc_update_timer(void *opaque)

{

    RTCState *s = opaque;

    int32_t irqs = REG_C_UF;

    int32_t new_irqs;



    assert((s->cmos_data[RTC_REG_A] & 0x60) != 0x60);



    /* UIP might have been latched, update time and clear it.  */

    rtc_update_time(s);

    s->cmos_data[RTC_REG_A] &= ~REG_A_UIP;



    if (check_alarm(s)) {

        irqs |= REG_C_AF;

        if (s->cmos_data[RTC_REG_B] & REG_B_AIE) {

            qemu_system_wakeup_request(QEMU_WAKEUP_REASON_RTC);

        }

    }

    new_irqs = irqs & ~s->cmos_data[RTC_REG_C];

    s->cmos_data[RTC_REG_C] |= irqs;

    if ((new_irqs & s->cmos_data[RTC_REG_B]) != 0) {

        s->cmos_data[RTC_REG_C] |= REG_C_IRQF;

        qemu_irq_raise(s->irq);

    }

    check_update_timer(s);

}
