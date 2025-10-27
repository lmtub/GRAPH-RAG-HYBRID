static void pl110_write(void *opaque, hwaddr offset,

                        uint64_t val, unsigned size)

{

    pl110_state *s = (pl110_state *)opaque;

    int n;



    /* For simplicity invalidate the display whenever a control register

       is written to.  */

    s->invalidate = 1;

    if (offset >= 0x200 && offset < 0x400) {

        /* Palette.  */

        n = (offset - 0x200) >> 2;

        s->raw_palette[(offset - 0x200) >> 2] = val;

        pl110_update_palette(s, n);

        return;

    }

    switch (offset >> 2) {

    case 0: /* LCDTiming0 */

        s->timing[0] = val;

        n = ((val & 0xfc) + 4) * 4;

        pl110_resize(s, n, s->rows);

        break;

    case 1: /* LCDTiming1 */

        s->timing[1] = val;

        n = (val & 0x3ff) + 1;

        pl110_resize(s, s->cols, n);

        break;

    case 2: /* LCDTiming2 */

        s->timing[2] = val;

        break;

    case 3: /* LCDTiming3 */

        s->timing[3] = val;

        break;

    case 4: /* LCDUPBASE */

        s->upbase = val;

        break;

    case 5: /* LCDLPBASE */

        s->lpbase = val;

        break;

    case 6: /* LCDIMSC */

        if (s->version != PL110) {

            goto control;

        }

    imsc:

        s->int_mask = val;

        pl110_update(s);

        break;

    case 7: /* LCDControl */

        if (s->version != PL110) {

            goto imsc;

        }

    control:

        s->cr = val;

        s->bpp = (val >> 1) & 7;

        if (pl110_enabled(s)) {

            qemu_console_resize(s->ds, s->cols, s->rows);

        }

        break;

    case 10: /* LCDICR */

        s->int_status &= ~val;

        pl110_update(s);

        break;

    default:

        hw_error("pl110_write: Bad offset %x\n", (int)offset);

    }

}
