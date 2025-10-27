static void display_mouse_set(DisplayChangeListener *dcl,

                              int x, int y, int on)

{

    SimpleSpiceDisplay *ssd = container_of(dcl, SimpleSpiceDisplay, dcl);



    qemu_mutex_lock(&ssd->lock);

    ssd->ptr_x = x;

    ssd->ptr_y = y;

    if (ssd->ptr_move) {

        g_free(ssd->ptr_move);

    }

    ssd->ptr_move = qemu_spice_create_cursor_update(ssd, NULL, on);

    qemu_mutex_unlock(&ssd->lock);

}
