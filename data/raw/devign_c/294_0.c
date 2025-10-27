static void handle_mousemotion(SDL_Event *ev)

{

    int max_x, max_y;

    struct sdl2_console *scon = get_scon_from_window(ev->key.windowID);



    if (qemu_input_is_absolute() || absolute_enabled) {

        int scr_w, scr_h;

        SDL_GetWindowSize(scon->real_window, &scr_w, &scr_h);

        max_x = scr_w - 1;

        max_y = scr_h - 1;

        if (gui_grab && (ev->motion.x == 0 || ev->motion.y == 0 ||

                         ev->motion.x == max_x || ev->motion.y == max_y)) {

            sdl_grab_end(scon);

        }

        if (!gui_grab &&

            (ev->motion.x > 0 && ev->motion.x < max_x &&

             ev->motion.y > 0 && ev->motion.y < max_y)) {

            sdl_grab_start(scon);

        }

    }

    if (gui_grab || qemu_input_is_absolute() || absolute_enabled) {

        sdl_send_mouse_event(scon, ev->motion.xrel, ev->motion.yrel,

                             ev->motion.x, ev->motion.y, ev->motion.state);

    }

}
