static void pl061_write(void *opaque, target_phys_addr_t offset,

                        uint32_t value)

{

    pl061_state *s = (pl061_state *)opaque;

    uint8_t mask;



    if (offset < 0x400) {

        mask = (offset >> 2) & s->dir;

        s->data = (s->data & ~mask) | (value & mask);

        pl061_update(s);

        return;

    }

    switch (offset) {

    case 0x400: /* Direction */

        s->dir = value;

        break;

    case 0x404: /* Interrupt sense */

        s->isense = value;

        break;

    case 0x408: /* Interrupt both edges */

        s->ibe = value;

        break;

    case 0x40c: /* Interrupt event */

        s->iev = value;

        break;

    case 0x410: /* Interrupt mask */

        s->im = value;

        break;

    case 0x41c: /* Interrupt clear */

        s->istate &= ~value;

        break;

    case 0x420: /* Alternate function select */

        mask = s->cr;

        s->afsel = (s->afsel & ~mask) | (value & mask);

        break;

    case 0x500: /* 2mA drive */

        s->dr2r = value;

        break;

    case 0x504: /* 4mA drive */

        s->dr4r = value;

        break;

    case 0x508: /* 8mA drive */

        s->dr8r = value;

        break;

    case 0x50c: /* Open drain */

        s->odr = value;

        break;

    case 0x510: /* Pull-up */

        s->pur = value;

        break;

    case 0x514: /* Pull-down */

        s->pdr = value;

        break;

    case 0x518: /* Slew rate control */

        s->slr = value;

        break;

    case 0x51c: /* Digital enable */

        s->den = value;

        break;

    case 0x520: /* Lock */

        s->locked = (value != 0xacce551);

        break;

    case 0x524: /* Commit */

        if (!s->locked)

            s->cr = value;

        break;

    default:

        hw_error("pl061_write: Bad offset %x\n", (int)offset);

    }

    pl061_update(s);

}
