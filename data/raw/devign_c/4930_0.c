static uint32_t check_alarm(RTCState *s)

{

    uint8_t alarm_hour, alarm_min, alarm_sec;

    uint8_t cur_hour, cur_min, cur_sec;



    alarm_sec = rtc_from_bcd(s, s->cmos_data[RTC_SECONDS_ALARM]);

    alarm_min = rtc_from_bcd(s, s->cmos_data[RTC_MINUTES_ALARM]);

    alarm_hour = rtc_from_bcd(s, s->cmos_data[RTC_HOURS_ALARM]);

    alarm_hour = convert_hour(s, alarm_hour);



    cur_sec = rtc_from_bcd(s, s->cmos_data[RTC_SECONDS]);

    cur_min = rtc_from_bcd(s, s->cmos_data[RTC_MINUTES]);

    cur_hour = rtc_from_bcd(s, s->cmos_data[RTC_HOURS]);

    cur_hour = convert_hour(s, cur_hour);



    if (((s->cmos_data[RTC_SECONDS_ALARM] & 0xc0) == 0xc0

                || alarm_sec == cur_sec) &&

            ((s->cmos_data[RTC_MINUTES_ALARM] & 0xc0) == 0xc0

             || alarm_min == cur_min) &&

            ((s->cmos_data[RTC_HOURS_ALARM] & 0xc0) == 0xc0

             || alarm_hour == cur_hour)) {

        return 1;

    }

    return 0;



}
