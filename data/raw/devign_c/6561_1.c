static void sdl_switch(DisplayChangeListener *dcl,

                       DisplaySurface *new_surface)

{

    PixelFormat pf = qemu_pixelformat_from_pixman(new_surface->format);



    /* temporary hack: allows to call sdl_switch to handle scaling changes */

    if (new_surface) {

        surface = new_surface;

    }



    if (!scaling_active) {

        do_sdl_resize(surface_width(surface), surface_height(surface), 0);

    } else if (real_screen->format->BitsPerPixel !=

               surface_bits_per_pixel(surface)) {

        do_sdl_resize(real_screen->w, real_screen->h,

                      surface_bits_per_pixel(surface));

    }



    if (guest_screen != NULL) {

        SDL_FreeSurface(guest_screen);

    }



#ifdef DEBUG_SDL

    printf("SDL: Creating surface with masks: %08x %08x %08x %08x\n",

           pf.rmask, pf.gmask, pf.bmask, pf.amask);

#endif



    guest_screen = SDL_CreateRGBSurfaceFrom

        (surface_data(surface),

         surface_width(surface), surface_height(surface),

         surface_bits_per_pixel(surface), surface_stride(surface),

         pf.rmask, pf.gmask,

         pf.bmask, pf.amask);

}
