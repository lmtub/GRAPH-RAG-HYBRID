static uint64_t pl061_read(void *opaque, hwaddr offset,

                           unsigned size)

{

    PL061State *s = (PL061State *)opaque;



    if (offset >= 0xfd0 && offset < 0x1000) {

        return s->id[(offset - 0xfd0) >> 2];

    }

    if (offset < 0x400) {

        return s->data & (offset >> 2);

    }

    switch (offset) {

    case 0x400: /* Direction */

        return s->dir;

    case 0x404: /* Interrupt sense */

        return s->isense;

    case 0x408: /* Interrupt both edges */

        return s->ibe;

    case 0x40c: /* Interrupt event */

        return s->iev;

    case 0x410: /* Interrupt mask */

        return s->im;

    case 0x414: /* Raw interrupt status */

        return s->istate;

    case 0x418: /* Masked interrupt status */

        return s->istate & s->im;

    case 0x420: /* Alternate function select */

        return s->afsel;

    case 0x500: /* 2mA drive */

        return s->dr2r;

    case 0x504: /* 4mA drive */

        return s->dr4r;

    case 0x508: /* 8mA drive */

        return s->dr8r;

    case 0x50c: /* Open drain */

        return s->odr;

    case 0x510: /* Pull-up */

        return s->pur;

    case 0x514: /* Pull-down */

        return s->pdr;

    case 0x518: /* Slew rate control */

        return s->slr;

    case 0x51c: /* Digital enable */

        return s->den;

    case 0x520: /* Lock */

        return s->locked;

    case 0x524: /* Commit */

        return s->cr;

    case 0x528: /* Analog mode select */

        return s->amsel;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "pl061_read: Bad offset %x\n", (int)offset);

        return 0;

    }

}
