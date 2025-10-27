CharDriverState *qemu_chr_open_msmouse(void)

{

    CharDriverState *chr;



    chr = g_malloc0(sizeof(CharDriverState));

    chr->chr_write = msmouse_chr_write;

    chr->chr_close = msmouse_chr_close;

    chr->explicit_be_open = true;



    qemu_add_mouse_event_handler(msmouse_event, chr, 0, "QEMU Microsoft Mouse");



    return chr;

}
