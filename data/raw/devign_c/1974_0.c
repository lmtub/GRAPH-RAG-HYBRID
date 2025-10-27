static void check_update_timer(RTCState *s)

{

    uint64_t next_update_time;

    uint64_t guest_nsec;

    int next_alarm_sec;



    /* From the data sheet: "Holding the dividers in reset prevents

     * interrupts from operating, while setting the SET bit allows"

     * them to occur.  However, it will prevent an alarm interrupt

     * from occurring, because the time of day is not updated.

     */

    if ((s->cmos_data[RTC_REG_A] & 0x60) == 0x60) {

        timer_del(s->update_timer);

        return;

    }

    if ((s->cmos_data[RTC_REG_C] & REG_C_UF) &&

        (s->cmos_data[RTC_REG_B] & REG_B_SET)) {

        timer_del(s->update_timer);

        return;

    }

    if ((s->cmos_data[RTC_REG_C] & REG_C_UF) &&

        (s->cmos_data[RTC_REG_C] & REG_C_AF)) {

        timer_del(s->update_timer);

        return;

    }



    guest_nsec = get_guest_rtc_ns(s) % NANOSECONDS_PER_SECOND;

    /* if UF is clear, reprogram to next second */

    next_update_time = qemu_clock_get_ns(rtc_clock)

        + NANOSECONDS_PER_SECOND - guest_nsec;



    /* Compute time of next alarm.  One second is already accounted

     * for in next_update_time.

     */

    next_alarm_sec = get_next_alarm(s);

    s->next_alarm_time = next_update_time +

                         (next_alarm_sec - 1) * NANOSECONDS_PER_SECOND;



    if (s->cmos_data[RTC_REG_C] & REG_C_UF) {

        /* UF is set, but AF is clear.  Program the timer to target

         * the alarm time.  */

        next_update_time = s->next_alarm_time;

    }

    if (next_update_time != timer_expire_time_ns(s->update_timer)) {

        timer_mod(s->update_timer, next_update_time);

    }

}
