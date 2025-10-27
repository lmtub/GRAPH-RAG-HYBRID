CharDriverState *text_console_init(DisplayState *ds, const char *p)

{

    CharDriverState *chr;

    TextConsole *s;

    unsigned width;

    unsigned height;

    static int color_inited;



    chr = qemu_mallocz(sizeof(CharDriverState));

    if (!chr)

        return NULL;

    s = new_console(ds, TEXT_CONSOLE);

    if (!s) {

        free(chr);

        return NULL;

    }

    if (!p)

        p = DEFAULT_MONITOR_SIZE;



    chr->opaque = s;

    chr->chr_write = console_puts;

    chr->chr_send_event = console_send_event;



    s->chr = chr;

    s->out_fifo.buf = s->out_fifo_buf;

    s->out_fifo.buf_size = sizeof(s->out_fifo_buf);

    s->kbd_timer = qemu_new_timer(rt_clock, kbd_send_chars, s);



    if (!color_inited) {

        color_inited = 1;

        console_color_init(s->ds);

    }

    s->y_displayed = 0;

    s->y_base = 0;

    s->total_height = DEFAULT_BACKSCROLL;

    s->x = 0;

    s->y = 0;

    width = s->ds->width;

    height = s->ds->height;

    if (p != 0) {

        width = strtoul(p, (char **)&p, 10);

        if (*p == 'C') {

            p++;

            width *= FONT_WIDTH;

        }

        if (*p == 'x') {

            p++;

            height = strtoul(p, (char **)&p, 10);

            if (*p == 'C') {

                p++;

                height *= FONT_HEIGHT;

            }

        }

    }

    s->g_width = width;

    s->g_height = height;



    s->hw_invalidate = text_console_invalidate;

    s->hw_text_update = text_console_update;

    s->hw = s;



    /* Set text attribute defaults */

    s->t_attrib_default.bold = 0;

    s->t_attrib_default.uline = 0;

    s->t_attrib_default.blink = 0;

    s->t_attrib_default.invers = 0;

    s->t_attrib_default.unvisible = 0;

    s->t_attrib_default.fgcol = COLOR_WHITE;

    s->t_attrib_default.bgcol = COLOR_BLACK;



    /* set current text attributes to default */

    s->t_attrib = s->t_attrib_default;

    text_console_resize(s);



    qemu_chr_reset(chr);



    return chr;

}
