static void handle_windowevent(SDL_Event *ev)

{

    struct sdl2_console *scon = get_scon_from_window(ev->window.windowID);



    if (!scon) {

        return;

    }



    switch (ev->window.event) {

    case SDL_WINDOWEVENT_RESIZED:

        {

            QemuUIInfo info;

            memset(&info, 0, sizeof(info));

            info.width = ev->window.data1;

            info.height = ev->window.data2;

            dpy_set_ui_info(scon->dcl.con, &info);

        }

        sdl2_redraw(scon);

        break;

    case SDL_WINDOWEVENT_EXPOSED:

        sdl2_redraw(scon);

        break;

    case SDL_WINDOWEVENT_FOCUS_GAINED:

    case SDL_WINDOWEVENT_ENTER:

        if (!gui_grab && (qemu_input_is_absolute() || absolute_enabled)) {

            absolute_mouse_grab(scon);

        }

        break;

    case SDL_WINDOWEVENT_FOCUS_LOST:

        if (gui_grab && !gui_fullscreen) {

            sdl_grab_end(scon);

        }

        break;

    case SDL_WINDOWEVENT_RESTORED:

        update_displaychangelistener(&scon->dcl, GUI_REFRESH_INTERVAL_DEFAULT);

        break;

    case SDL_WINDOWEVENT_MINIMIZED:

        update_displaychangelistener(&scon->dcl, 500);

        break;

    case SDL_WINDOWEVENT_CLOSE:

        if (qemu_console_is_graphic(scon->dcl.con)) {

            if (!no_quit) {

                no_shutdown = 0;

                qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI);

            }

        } else {

            SDL_HideWindow(scon->real_window);

            scon->hidden = true;

        }

        break;

    case SDL_WINDOWEVENT_SHOWN:

        if (scon->hidden) {

            SDL_HideWindow(scon->real_window);

        }

        break;

    case SDL_WINDOWEVENT_HIDDEN:

        if (!scon->hidden) {

            SDL_ShowWindow(scon->real_window);

        }

        break;

    }

}
