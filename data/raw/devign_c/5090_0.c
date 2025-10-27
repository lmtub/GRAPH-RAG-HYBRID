static void cadence_ttc_write(void *opaque, target_phys_addr_t offset,

        uint64_t value, unsigned size)

{

    CadenceTimerState *s = cadence_timer_from_addr(opaque, offset);



    DB_PRINT("addr: %08x data %08x\n", offset, (unsigned)value);



    cadence_timer_sync(s);



    switch (offset) {

    case 0x00: /* clock control */

    case 0x04:

    case 0x08:

        s->reg_clock = value & 0x3F;

        break;



    case 0x0c: /* counter control */

    case 0x10:

    case 0x14:

        if (value & COUNTER_CTRL_RST) {

            s->reg_value = 0;

        }

        s->reg_count = value & 0x3f & ~COUNTER_CTRL_RST;

        break;



    case 0x24: /* interval register */

    case 0x28:

    case 0x2c:

        s->reg_interval = value & 0xffff;

        break;



    case 0x30: /* match register */

    case 0x34:

    case 0x38:

        s->reg_match[0] = value & 0xffff;



    case 0x3c: /* match register */

    case 0x40:

    case 0x44:

        s->reg_match[1] = value & 0xffff;



    case 0x48: /* match register */

    case 0x4c:

    case 0x50:

        s->reg_match[2] = value & 0xffff;

        break;



    case 0x54: /* interrupt register */

    case 0x58:

    case 0x5c:

        break;



    case 0x60: /* interrupt enable */

    case 0x64:

    case 0x68:

        s->reg_intr_en = value & 0x3f;

        break;



    case 0x6c: /* event control */

    case 0x70:

    case 0x74:

        s->reg_event_ctrl = value & 0x07;

        break;



    default:

        return;

    }



    cadence_timer_run(s);

    cadence_timer_update(s);

}
