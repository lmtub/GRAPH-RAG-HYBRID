static void pl061_write(void *opaque, hwaddr offset,

                        uint64_t value, unsigned size)

{

    PL061State *s = (PL061State *)opaque;

    uint8_t mask;



    if (offset < 0x400) {

        mask = (offset >> 2) & s->dir;

        s->data = (s->data & ~mask) | (value & mask);

        pl061_update(s);

        return;

    }

    switch (offset) {

    case 0x400: /* Direction */

        s->dir = value & 0xff;

        break;

    case 0x404: /* Interrupt sense */

        s->isense = value & 0xff;

        break;

    case 0x408: /* Interrupt both edges */

        s->ibe = value & 0xff;

        break;

    case 0x40c: /* Interrupt event */

        s->iev = value & 0xff;

        break;

    case 0x410: /* Interrupt mask */

        s->im = value & 0xff;

        break;

    case 0x41c: /* Interrupt clear */

        s->istate &= ~value;

        break;

    case 0x420: /* Alternate function select */

        mask = s->cr;

        s->afsel = (s->afsel & ~mask) | (value & mask);

        break;

    case 0x500: /* 2mA drive */

        s->dr2r = value & 0xff;

        break;

    case 0x504: /* 4mA drive */

        s->dr4r = value & 0xff;

        break;

    case 0x508: /* 8mA drive */

        s->dr8r = value & 0xff;

        break;

    case 0x50c: /* Open drain */

        s->odr = value & 0xff;

        break;

    case 0x510: /* Pull-up */

        s->pur = value & 0xff;

        break;

    case 0x514: /* Pull-down */

        s->pdr = value & 0xff;

        break;

    case 0x518: /* Slew rate control */

        s->slr = value & 0xff;

        break;

    case 0x51c: /* Digital enable */

        s->den = value & 0xff;

        break;

    case 0x520: /* Lock */

        s->locked = (value != 0xacce551);

        break;

    case 0x524: /* Commit */

        if (!s->locked)

            s->cr = value & 0xff;

        break;

    case 0x528:

        s->amsel = value & 0xff;

        break;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "pl061_write: Bad offset %x\n", (int)offset);

    }

    pl061_update(s);

}
