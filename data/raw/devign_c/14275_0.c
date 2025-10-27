QemuConsole *graphic_console_init(graphic_hw_update_ptr update,

                                  graphic_hw_invalidate_ptr invalidate,

                                  graphic_hw_screen_dump_ptr screen_dump,

                                  graphic_hw_text_update_ptr text_update,

                                  void *opaque)

{

    int width = 640;

    int height = 480;

    QemuConsole *s;

    DisplayState *ds;



    ds = get_alloc_displaystate();

    trace_console_gfx_new();

    s = new_console(ds, GRAPHIC_CONSOLE);

    s->hw_update = update;

    s->hw_invalidate = invalidate;

    s->hw_screen_dump = screen_dump;

    s->hw_text_update = text_update;

    s->hw = opaque;



    s->surface = qemu_create_displaysurface(width, height);

    return s;

}
