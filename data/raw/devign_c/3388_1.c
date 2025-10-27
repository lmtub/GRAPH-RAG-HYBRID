static CharDriverState *text_console_init(ChardevVC *vc)

{

    CharDriverState *chr;

    QemuConsole *s;

    unsigned width = 0;

    unsigned height = 0;



    chr = g_malloc0(sizeof(CharDriverState));



    if (vc->has_width) {

        width = vc->width;

    } else if (vc->has_cols) {

        width = vc->cols * FONT_WIDTH;

    }



    if (vc->has_height) {

        height = vc->height;

    } else if (vc->has_rows) {

        height = vc->rows * FONT_HEIGHT;

    }



    trace_console_txt_new(width, height);

    if (width == 0 || height == 0) {

        s = new_console(NULL, TEXT_CONSOLE);

    } else {

        s = new_console(NULL, TEXT_CONSOLE_FIXED_SIZE);

        s->surface = qemu_create_displaysurface(width, height);

    }



    if (!s) {

        g_free(chr);

        return NULL;

    }



    s->chr = chr;

    chr->opaque = s;

    chr->chr_set_echo = text_console_set_echo;

    /* console/chardev init sometimes completes elsewhere in a 2nd

     * stage, so defer OPENED events until they are fully initialized

     */

    chr->explicit_be_open = true;



    if (display_state) {

        text_console_do_init(chr, display_state);

    }

    return chr;

}
