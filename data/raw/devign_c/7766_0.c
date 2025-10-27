static DisplayType select_display(const char *p)

{

    Error *err = NULL;

    const char *opts;

    DisplayType display = DT_DEFAULT;



    if (strstart(p, "sdl", &opts)) {

#ifdef CONFIG_SDL

        display = DT_SDL;

        while (*opts) {

            const char *nextopt;



            if (strstart(opts, ",frame=", &nextopt)) {

                opts = nextopt;

                if (strstart(opts, "on", &nextopt)) {

                    no_frame = 0;

                } else if (strstart(opts, "off", &nextopt)) {

                    no_frame = 1;

                } else {

                    goto invalid_sdl_args;

                }

            } else if (strstart(opts, ",alt_grab=", &nextopt)) {

                opts = nextopt;

                if (strstart(opts, "on", &nextopt)) {

                    alt_grab = 1;

                } else if (strstart(opts, "off", &nextopt)) {

                    alt_grab = 0;

                } else {

                    goto invalid_sdl_args;

                }

            } else if (strstart(opts, ",ctrl_grab=", &nextopt)) {

                opts = nextopt;

                if (strstart(opts, "on", &nextopt)) {

                    ctrl_grab = 1;

                } else if (strstart(opts, "off", &nextopt)) {

                    ctrl_grab = 0;

                } else {

                    goto invalid_sdl_args;

                }

            } else if (strstart(opts, ",window_close=", &nextopt)) {

                opts = nextopt;

                if (strstart(opts, "on", &nextopt)) {

                    no_quit = 0;

                } else if (strstart(opts, "off", &nextopt)) {

                    no_quit = 1;

                } else {

                    goto invalid_sdl_args;

                }

            } else if (strstart(opts, ",gl=", &nextopt)) {

                opts = nextopt;

                if (strstart(opts, "on", &nextopt)) {

                    request_opengl = 1;

                } else if (strstart(opts, "off", &nextopt)) {

                    request_opengl = 0;

                } else {

                    goto invalid_sdl_args;

                }

            } else {

            invalid_sdl_args:

                fprintf(stderr, "Invalid SDL option string: %s\n", p);

                exit(1);

            }

            opts = nextopt;

        }

#else

        fprintf(stderr, "SDL support is disabled\n");

        exit(1);

#endif

    } else if (strstart(p, "vnc", &opts)) {

#ifdef CONFIG_VNC

        if (*opts == '=') {

            if (vnc_parse(opts + 1, &err) == NULL) {

                error_report_err(err);

                exit(1);

            }

        } else {

            fprintf(stderr, "VNC requires a display argument vnc=<display>\n");

            exit(1);

        }

#else

        fprintf(stderr, "VNC support is disabled\n");

        exit(1);

#endif

    } else if (strstart(p, "curses", &opts)) {

#ifdef CONFIG_CURSES

        display = DT_CURSES;

#else

        fprintf(stderr, "Curses support is disabled\n");

        exit(1);

#endif

    } else if (strstart(p, "gtk", &opts)) {

#ifdef CONFIG_GTK

        display = DT_GTK;

        while (*opts) {

            const char *nextopt;



            if (strstart(opts, ",grab_on_hover=", &nextopt)) {

                opts = nextopt;

                if (strstart(opts, "on", &nextopt)) {

                    grab_on_hover = true;

                } else if (strstart(opts, "off", &nextopt)) {

                    grab_on_hover = false;

                } else {

                    goto invalid_gtk_args;

                }

            } else if (strstart(opts, ",gl=", &nextopt)) {

                opts = nextopt;

                if (strstart(opts, "on", &nextopt)) {

                    request_opengl = 1;

                } else if (strstart(opts, "off", &nextopt)) {

                    request_opengl = 0;

                } else {

                    goto invalid_gtk_args;

                }

            } else {

            invalid_gtk_args:

                fprintf(stderr, "Invalid GTK option string: %s\n", p);

                exit(1);

            }

            opts = nextopt;

        }

#else

        fprintf(stderr, "GTK support is disabled\n");

        exit(1);

#endif

    } else if (strstart(p, "none", &opts)) {

        display = DT_NONE;

    } else {

        fprintf(stderr, "Unknown display type: %s\n", p);

        exit(1);

    }



    return display;

}
