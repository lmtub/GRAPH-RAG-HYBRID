CharDriverState *text_console_init(QemuOpts *opts)

{

    CharDriverState *chr;

    QemuConsole *s;

    unsigned width;

    unsigned height;



    chr = g_malloc0(sizeof(CharDriverState));



    width = qemu_opt_get_number(opts, "width", 0);

    if (width == 0)

        width = qemu_opt_get_number(opts, "cols", 0) * FONT_WIDTH;



    height = qemu_opt_get_number(opts, "height", 0);

    if (height == 0)

        height = qemu_opt_get_number(opts, "rows", 0) * FONT_HEIGHT;



    if (width == 0 || height == 0) {

        s = new_console(NULL, TEXT_CONSOLE);

    } else {

        s = new_console(NULL, TEXT_CONSOLE_FIXED_SIZE);

    }



    if (!s) {

        g_free(chr);

        return NULL;

    }



    s->chr = chr;

    s->g_width = width;

    s->g_height = height;

    chr->opaque = s;

    chr->chr_set_echo = text_console_set_echo;

    return chr;

}
