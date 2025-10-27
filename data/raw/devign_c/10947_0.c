void sdl_display_init(DisplayState *ds, int full_screen, int no_frame)

{

    int flags;

    uint8_t data = 0;



#if defined(__APPLE__)

    /* always use generic keymaps */

    if (!keyboard_layout)

        keyboard_layout = "en-us";

#endif

    if(keyboard_layout) {

        kbd_layout = init_keyboard_layout(keyboard_layout);

        if (!kbd_layout)

            exit(1);

    }



    if (no_frame)

        gui_noframe = 1;



    flags = SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE;

    if (SDL_Init (flags)) {

        fprintf(stderr, "Could not initialize SDL - exiting\n");

        exit(1);

    }

#ifndef _WIN32

    /* NOTE: we still want Ctrl-C to work, so we undo the SDL redirections */

    signal(SIGINT, SIG_DFL);

    signal(SIGQUIT, SIG_DFL);

#endif



    ds->dpy_update = sdl_update;

    ds->dpy_resize = sdl_resize;

    ds->dpy_refresh = sdl_refresh;

    ds->dpy_fill = sdl_fill;

    ds->mouse_set = sdl_mouse_warp;

    ds->cursor_define = sdl_mouse_define;



    sdl_resize(ds, 640, 400);

    sdl_update_caption();

    SDL_EnableKeyRepeat(250, 50);

    gui_grab = 0;



    sdl_cursor_hidden = SDL_CreateCursor(&data, &data, 8, 1, 0, 0);

    sdl_cursor_normal = SDL_GetCursor();



    atexit(sdl_cleanup);

    if (full_screen) {

        gui_fullscreen = 1;

        gui_fullscreen_initial_grab = 1;

        sdl_grab_start();

    }

}
