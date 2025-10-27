static int video_open(VideoState *is){

    int flags = SDL_HWSURFACE|SDL_ASYNCBLIT|SDL_HWACCEL;

    int w,h;



    if(is_full_screen) flags |= SDL_FULLSCREEN;

    else               flags |= SDL_RESIZABLE;



    if (is_full_screen && fs_screen_width) {

        w = fs_screen_width;

        h = fs_screen_height;

    } else if(!is_full_screen && screen_width){

        w = screen_width;

        h = screen_height;

    }else if (is->video_st && is->video_st->codec->width){

        w = is->video_st->codec->width;

        h = is->video_st->codec->height;

    } else {

        w = 640;

        h = 480;

    }

#ifndef SYS_DARWIN

    screen = SDL_SetVideoMode(w, h, 0, flags);

#else

    /* setting bits_per_pixel = 0 or 32 causes blank video on OS X */

    screen = SDL_SetVideoMode(w, h, 24, flags);

#endif

    if (!screen) {

        fprintf(stderr, "SDL: could not set video mode - exiting\n");

        return -1;

    }

    SDL_WM_SetCaption("FFplay", "FFplay");



    is->width = screen->w;

    is->height = screen->h;



    return 0;

}
