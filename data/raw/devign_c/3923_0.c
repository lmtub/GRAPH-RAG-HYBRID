static void pxa2xx_rtc_write(void *opaque, hwaddr addr,

                             uint64_t value64, unsigned size)

{

    PXA2xxRTCState *s = (PXA2xxRTCState *) opaque;

    uint32_t value = value64;



    switch (addr) {

    case RTTR:

        if (!(s->rttr & (1U << 31))) {

            pxa2xx_rtc_hzupdate(s);

            s->rttr = value;

            pxa2xx_rtc_alarm_update(s, s->rtsr);

        }

        break;



    case RTSR:

        if ((s->rtsr ^ value) & (1 << 15))

            pxa2xx_rtc_piupdate(s);



        if ((s->rtsr ^ value) & (1 << 12))

            pxa2xx_rtc_swupdate(s);



        if (((s->rtsr ^ value) & 0x4aac) | (value & ~0xdaac))

            pxa2xx_rtc_alarm_update(s, value);



        s->rtsr = (value & 0xdaac) | (s->rtsr & ~(value & ~0xdaac));

        pxa2xx_rtc_int_update(s);

        break;



    case RTAR:

        s->rtar = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RDAR1:

        s->rdar1 = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RDAR2:

        s->rdar2 = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RYAR1:

        s->ryar1 = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RYAR2:

        s->ryar2 = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case SWAR1:

        pxa2xx_rtc_swupdate(s);

        s->swar1 = value;

        s->last_swcr = 0;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case SWAR2:

        s->swar2 = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case PIAR:

        s->piar = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RCNR:

        pxa2xx_rtc_hzupdate(s);

        s->last_rcnr = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RDCR:

        pxa2xx_rtc_hzupdate(s);

        s->last_rdcr = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RYCR:

        s->last_rycr = value;

        break;



    case SWCR:

        pxa2xx_rtc_swupdate(s);

        s->last_swcr = value;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    case RTCPICR:

        pxa2xx_rtc_piupdate(s);

        s->last_rtcpicr = value & 0xffff;

        pxa2xx_rtc_alarm_update(s, s->rtsr);

        break;



    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

    }

}
