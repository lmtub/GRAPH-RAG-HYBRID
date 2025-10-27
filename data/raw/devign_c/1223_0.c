static void absolute_mouse_grab(void)

{

    int mouse_x, mouse_y;



    if (SDL_GetAppState() & SDL_APPINPUTFOCUS) {

        SDL_GetMouseState(&mouse_x, &mouse_y);

        if (mouse_x > 0 && mouse_x < real_screen->w - 1 &&

            mouse_y > 0 && mouse_y < real_screen->h - 1) {

            sdl_grab_start();

        }

    }

}
