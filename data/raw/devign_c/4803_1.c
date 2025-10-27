static void handle_keydown(SDL_Event *ev)
{
    int mod_state, win;
    struct sdl2_console *scon = get_scon_from_window(ev->key.windowID);
    if (alt_grab) {
        mod_state = (SDL_GetModState() & (gui_grab_code | KMOD_LSHIFT)) ==
            (gui_grab_code | KMOD_LSHIFT);
    } else if (ctrl_grab) {
        mod_state = (SDL_GetModState() & KMOD_RCTRL) == KMOD_RCTRL;
    } else {
        mod_state = (SDL_GetModState() & gui_grab_code) == gui_grab_code;
    gui_key_modifier_pressed = mod_state;
    if (gui_key_modifier_pressed) {
        switch (ev->key.keysym.scancode) {
        case SDL_SCANCODE_2:
        case SDL_SCANCODE_3:
        case SDL_SCANCODE_4:
        case SDL_SCANCODE_5:
        case SDL_SCANCODE_6:
        case SDL_SCANCODE_7:
        case SDL_SCANCODE_8:
        case SDL_SCANCODE_9:
            win = ev->key.keysym.scancode - SDL_SCANCODE_1;
            if (win < sdl2_num_outputs) {
                sdl2_console[win].hidden = !sdl2_console[win].hidden;
                if (sdl2_console[win].real_window) {
                    if (sdl2_console[win].hidden) {
                        SDL_HideWindow(sdl2_console[win].real_window);
                    } else {
                        SDL_ShowWindow(sdl2_console[win].real_window);
                gui_keysym = 1;
            break;
        case SDL_SCANCODE_F:
            toggle_full_screen(scon);
            gui_keysym = 1;
            break;
        case SDL_SCANCODE_U:
            sdl2_window_destroy(scon);
            sdl2_window_create(scon);
            if (!scon->opengl) {
                /* re-create scon->texture */
                sdl2_2d_switch(&scon->dcl, scon->surface);
            gui_keysym = 1;
            break;
#if 0
        case SDL_SCANCODE_KP_PLUS:
        case SDL_SCANCODE_KP_MINUS:
            if (!gui_fullscreen) {
                int scr_w, scr_h;
                int width, height;
                SDL_GetWindowSize(scon->real_window, &scr_w, &scr_h);
                width = MAX(scr_w + (ev->key.keysym.scancode ==
                                     SDL_SCANCODE_KP_PLUS ? 50 : -50),
                            160);
                height = (surface_height(scon->surface) * width) /
                    surface_width(scon->surface);
                fprintf(stderr, "%s: scale to %dx%d\n",
                        __func__, width, height);
                sdl_scale(scon, width, height);
                sdl2_redraw(scon);
                gui_keysym = 1;
#endif
        default:
            break;
    if (!gui_keysym) {
        sdl2_process_key(scon, &ev->key);