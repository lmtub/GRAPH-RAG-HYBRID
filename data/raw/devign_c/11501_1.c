static void msmouse_chr_close (struct CharDriverState *chr)

{

    MouseState *mouse = chr->opaque;



    qemu_input_handler_unregister(mouse->hs);

    g_free(mouse);

    g_free(chr);

}
