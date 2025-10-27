static void sdl_resize(DisplayState *ds, int w, int h)

{

    int flags;



    //    printf("resizing to %d %d\n", w, h);



    flags = SDL_HWSURFACE|SDL_ASYNCBLIT|SDL_HWACCEL;

    flags |= SDL_RESIZABLE;

    if (gui_fullscreen)

        flags |= SDL_FULLSCREEN;

    screen = SDL_SetVideoMode(w, h, 0, flags);

    if (!screen) {

        fprintf(stderr, "Could not open SDL display\n");

        exit(1);

    }

    ds->data = screen->pixels;

    ds->linesize = screen->pitch;

    ds->depth = screen->format->BitsPerPixel;

    ds->width = w;

    ds->height = h;

}
