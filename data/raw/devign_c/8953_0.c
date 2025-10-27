static void exynos4210_rtc_write(void *opaque, target_phys_addr_t offset,

        uint64_t value, unsigned size)

{

    Exynos4210RTCState *s = (Exynos4210RTCState *)opaque;



    switch (offset) {

    case INTP:

        if (value & INTP_ALM_ENABLE) {

            qemu_irq_lower(s->alm_irq);

            s->reg_intp &= (~INTP_ALM_ENABLE);

        }

        if (value & INTP_TICK_ENABLE) {

            qemu_irq_lower(s->tick_irq);

            s->reg_intp &= (~INTP_TICK_ENABLE);

        }

        break;

    case RTCCON:

        if (value & RTC_ENABLE) {

            exynos4210_rtc_update_freq(s, value);

        }

        if ((value & RTC_ENABLE) > (s->reg_rtccon & RTC_ENABLE)) {

            /* clock timer */

            ptimer_set_count(s->ptimer_1Hz, RTC_BASE_FREQ);

            ptimer_run(s->ptimer_1Hz, 1);

            DPRINTF("run clock timer\n");

        }

        if ((value & RTC_ENABLE) < (s->reg_rtccon & RTC_ENABLE)) {

            /* tick timer */

            ptimer_stop(s->ptimer);

            /* clock timer */

            ptimer_stop(s->ptimer_1Hz);

            DPRINTF("stop all timers\n");

        }

        if (value & RTC_ENABLE) {

            if ((value & TICK_TIMER_ENABLE) >

                (s->reg_rtccon & TICK_TIMER_ENABLE) &&

                (s->reg_ticcnt)) {

                ptimer_set_count(s->ptimer, s->reg_ticcnt);

                ptimer_run(s->ptimer, 1);

                DPRINTF("run tick timer\n");

            }

            if ((value & TICK_TIMER_ENABLE) <

                (s->reg_rtccon & TICK_TIMER_ENABLE)) {

                ptimer_stop(s->ptimer);

            }

        }

        s->reg_rtccon = value;

        break;

    case TICCNT:

        if (value > TICNT_THRESHHOLD) {

            s->reg_ticcnt = value;

        } else {

            fprintf(stderr,

                    "[exynos4210.rtc: bad TICNT value %u ]\n",

                    (uint32_t)value);

        }

        break;



    case RTCALM:

        s->reg_rtcalm = value;

        break;

    case ALMSEC:

        s->reg_almsec = (value & 0x7f);

        break;

    case ALMMIN:

        s->reg_almmin = (value & 0x7f);

        break;

    case ALMHOUR:

        s->reg_almhour = (value & 0x3f);

        break;

    case ALMDAY:

        s->reg_almday = (value & 0x3f);

        break;

    case ALMMON:

        s->reg_almmon = (value & 0x1f);

        break;

    case ALMYEAR:

        s->reg_almyear = (value & 0x0fff);

        break;



    case BCDSEC:

        if (s->reg_rtccon & RTC_ENABLE) {

            s->current_tm.tm_sec = (int)from_bcd((uint8_t)value);

        }

        break;

    case BCDMIN:

        if (s->reg_rtccon & RTC_ENABLE) {

            s->current_tm.tm_min = (int)from_bcd((uint8_t)value);

        }

        break;

    case BCDHOUR:

        if (s->reg_rtccon & RTC_ENABLE) {

            s->current_tm.tm_hour = (int)from_bcd((uint8_t)value);

        }

        break;

    case BCDDAYWEEK:

        if (s->reg_rtccon & RTC_ENABLE) {

            s->current_tm.tm_wday = (int)from_bcd((uint8_t)value);

        }

        break;

    case BCDDAY:

        if (s->reg_rtccon & RTC_ENABLE) {

            s->current_tm.tm_mday = (int)from_bcd((uint8_t)value);

        }

        break;

    case BCDMON:

        if (s->reg_rtccon & RTC_ENABLE) {

            s->current_tm.tm_mon = (int)from_bcd((uint8_t)value) - 1;

        }

        break;

    case BCDYEAR:

        if (s->reg_rtccon & RTC_ENABLE) {

            /* 3 digits */

            s->current_tm.tm_year = (int)from_bcd((uint8_t)value) +

                    (int)from_bcd((uint8_t)((value >> 8) & 0x0f)) * 100;

        }

        break;



    default:

        fprintf(stderr,

                "[exynos4210.rtc: bad write offset " TARGET_FMT_plx "]\n",

                offset);

        break;



    }

}
