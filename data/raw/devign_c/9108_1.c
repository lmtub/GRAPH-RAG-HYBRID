static void do_sdl_resize(int new_width, int new_height, int bpp)

{

    int flags;



    //    printf("resizing to %d %d\n", w, h);



    flags = SDL_HWSURFACE|SDL_ASYNCBLIT|SDL_HWACCEL|SDL_RESIZABLE;

    if (gui_fullscreen)

        flags |= SDL_FULLSCREEN;

    if (gui_noframe)

        flags |= SDL_NOFRAME;



    width = new_width;

    height = new_height;

    real_screen = SDL_SetVideoMode(width, height, bpp, flags);

    if (!real_screen) {

	fprintf(stderr, "Could not open SDL display (%dx%dx%d): %s\n", width, 

		height, bpp, SDL_GetError());

        exit(1);

    }

}
