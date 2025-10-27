static uint32_t cadence_ttc_read_imp(void *opaque, target_phys_addr_t offset)

{

    CadenceTimerState *s = cadence_timer_from_addr(opaque, offset);

    uint32_t value;



    cadence_timer_sync(s);

    cadence_timer_run(s);



    switch (offset) {

    case 0x00: /* clock control */

    case 0x04:

    case 0x08:

        return s->reg_clock;



    case 0x0c: /* counter control */

    case 0x10:

    case 0x14:

        return s->reg_count;



    case 0x18: /* counter value */

    case 0x1c:

    case 0x20:

        return (uint16_t)(s->reg_value >> 16);



    case 0x24: /* reg_interval counter */

    case 0x28:

    case 0x2c:

        return s->reg_interval;



    case 0x30: /* match 1 counter */

    case 0x34:

    case 0x38:

        return s->reg_match[0];



    case 0x3c: /* match 2 counter */

    case 0x40:

    case 0x44:

        return s->reg_match[1];



    case 0x48: /* match 3 counter */

    case 0x4c:

    case 0x50:

        return s->reg_match[2];



    case 0x54: /* interrupt register */

    case 0x58:

    case 0x5c:

        /* cleared after read */

        value = s->reg_intr;

        s->reg_intr = 0;

        cadence_timer_update(s);

        return value;



    case 0x60: /* interrupt enable */

    case 0x64:

    case 0x68:

        return s->reg_intr_en;



    case 0x6c:

    case 0x70:

    case 0x74:

        return s->reg_event_ctrl;



    case 0x78:

    case 0x7c:

    case 0x80:

        return s->reg_event;



    default:

        return 0;

    }

}
