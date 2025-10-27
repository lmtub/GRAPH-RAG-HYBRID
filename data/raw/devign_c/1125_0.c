static uint64_t exynos4210_rtc_read(void *opaque, target_phys_addr_t offset,

        unsigned size)

{

    uint32_t value = 0;

    Exynos4210RTCState *s = (Exynos4210RTCState *)opaque;



    switch (offset) {

    case INTP:

        value = s->reg_intp;

        break;

    case RTCCON:

        value = s->reg_rtccon;

        break;

    case TICCNT:

        value = s->reg_ticcnt;

        break;

    case RTCALM:

        value = s->reg_rtcalm;

        break;

    case ALMSEC:

        value = s->reg_almsec;

        break;

    case ALMMIN:

        value = s->reg_almmin;

        break;

    case ALMHOUR:

        value = s->reg_almhour;

        break;

    case ALMDAY:

        value = s->reg_almday;

        break;

    case ALMMON:

        value = s->reg_almmon;

        break;

    case ALMYEAR:

        value = s->reg_almyear;

        break;



    case BCDSEC:

        value = (uint32_t)to_bcd((uint8_t)s->current_tm.tm_sec);

        break;

    case BCDMIN:

        value = (uint32_t)to_bcd((uint8_t)s->current_tm.tm_min);

        break;

    case BCDHOUR:

        value = (uint32_t)to_bcd((uint8_t)s->current_tm.tm_hour);

        break;

    case BCDDAYWEEK:

        value = (uint32_t)to_bcd((uint8_t)s->current_tm.tm_wday);

        break;

    case BCDDAY:

        value = (uint32_t)to_bcd((uint8_t)s->current_tm.tm_mday);

        break;

    case BCDMON:

        value = (uint32_t)to_bcd((uint8_t)s->current_tm.tm_mon + 1);

        break;

    case BCDYEAR:

        value = BCD3DIGITS(s->current_tm.tm_year);

        break;



    case CURTICNT:

        s->reg_curticcnt = ptimer_get_count(s->ptimer);

        value = s->reg_curticcnt;

        break;



    default:

        fprintf(stderr,

                "[exynos4210.rtc: bad read offset " TARGET_FMT_plx "]\n",

                offset);

        break;

    }

    return value;

}
