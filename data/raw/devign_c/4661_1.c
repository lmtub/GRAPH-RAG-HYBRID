static int video_open(VideoState *is)

{

    int w,h;



    if (screen_width) {

        w = screen_width;

        h = screen_height;

    } else {

        w = default_width;

        h = default_height;

    }



    if (!window) {

        int flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;

        if (!window_title)

            window_title = input_filename;

        if (is_full_screen)

            flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;



        window = SDL_CreateWindow(window_title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, w, h, flags);

        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");

        if (window) {

            SDL_RendererInfo info;

            renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

            if (!renderer) {

                av_log(NULL, AV_LOG_WARNING, "Failed to initialize a hardware accelerated renderer: %s\n", SDL_GetError());

                renderer = SDL_CreateRenderer(window, -1, 0);

            }

            if (renderer) {

                if (!SDL_GetRendererInfo(renderer, &info))

                    av_log(NULL, AV_LOG_VERBOSE, "Initialized %s renderer.\n", info.name);

            }

        }

    } else {

        SDL_SetWindowSize(window, w, h);

    }



    if (!window || !renderer) {

        av_log(NULL, AV_LOG_FATAL, "SDL: could not set video mode - exiting\n");

        do_exit(is);

    }



    is->width  = w;

    is->height = h;



    return 0;

}