static void text_console_do_init(CharDriverState *chr, DisplayState *ds)

{

    TextConsole *s;

    static int color_inited;



    s = chr->opaque;



    chr->chr_write = console_puts;

    chr->chr_send_event = console_send_event;



    s->out_fifo.buf = s->out_fifo_buf;

    s->out_fifo.buf_size = sizeof(s->out_fifo_buf);

    s->kbd_timer = qemu_new_timer(rt_clock, kbd_send_chars, s);

    s->ds = ds;



    if (!color_inited) {

        color_inited = 1;

        console_color_init(s->ds);

    }

    s->y_displayed = 0;

    s->y_base = 0;

    s->total_height = DEFAULT_BACKSCROLL;

    s->x = 0;

    s->y = 0;

    if (s->console_type == TEXT_CONSOLE) {

        s->g_width = ds_get_width(s->ds);

        s->g_height = ds_get_height(s->ds);

    }



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



    if (chr->label) {

        char msg[128];

        int len;



        s->t_attrib.bgcol = COLOR_BLUE;

        len = snprintf(msg, sizeof(msg), "%s console\r\n", chr->label);

        console_puts(chr, (uint8_t*)msg, len);

        s->t_attrib = s->t_attrib_default;

    }



    qemu_chr_generic_open(chr);

    if (chr->init)

        chr->init(chr);

}
