static void handle_mousewheel(SDL_Event *ev)

{

    struct sdl2_console *scon = get_scon_from_window(ev->key.windowID);

    SDL_MouseWheelEvent *wev = &ev->wheel;

    InputButton btn;



    if (wev->y > 0) {

        btn = INPUT_BUTTON_WHEEL_UP;

    } else if (wev->y < 0) {

        btn = INPUT_BUTTON_WHEEL_DOWN;

    } else {

        return;

    }



    qemu_input_queue_btn(scon->dcl.con, btn, true);

    qemu_input_event_sync();

    qemu_input_queue_btn(scon->dcl.con, btn, false);

    qemu_input_event_sync();

}
