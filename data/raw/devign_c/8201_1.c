void curses_display_init(DisplayState *ds, int full_screen)

{

#ifndef _WIN32

    if (!isatty(1)) {

        fprintf(stderr, "We need a terminal output\n");

        exit(1);

    }

#endif



    curses_setup();

    curses_keyboard_setup();

    atexit(curses_atexit);



    curses_winch_init();



    dcl = (DisplayChangeListener *) g_malloc0(sizeof(DisplayChangeListener));

    dcl->ops = &dcl_ops;

    register_displaychangelistener(dcl);



    invalidate = 1;

}
