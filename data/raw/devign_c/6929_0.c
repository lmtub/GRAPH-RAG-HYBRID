static void sdl_mouse_warp(int x, int y, int on)

{

    if (on) {

        if (!guest_cursor)

            sdl_show_cursor();

        if (gui_grab || kbd_mouse_is_absolute() || absolute_enabled) {

            SDL_SetCursor(guest_sprite);

            SDL_WarpMouse(x, y);

        }

    } else if (gui_grab)

        sdl_hide_cursor();

    guest_cursor = on;

    guest_x = x, guest_y = y;

}
