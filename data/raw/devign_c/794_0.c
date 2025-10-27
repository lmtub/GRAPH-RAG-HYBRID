static void sdl_send_mouse_event(int dx, int dy, int x, int y, int state)

{

    static uint32_t bmap[INPUT_BUTTON__MAX] = {

        [INPUT_BUTTON_LEFT]       = SDL_BUTTON(SDL_BUTTON_LEFT),

        [INPUT_BUTTON_MIDDLE]     = SDL_BUTTON(SDL_BUTTON_MIDDLE),

        [INPUT_BUTTON_RIGHT]      = SDL_BUTTON(SDL_BUTTON_RIGHT),

        [INPUT_BUTTON_WHEEL_UP]   = SDL_BUTTON(SDL_BUTTON_WHEELUP),

        [INPUT_BUTTON_WHEEL_DOWN] = SDL_BUTTON(SDL_BUTTON_WHEELDOWN),

    };

    static uint32_t prev_state;



    if (prev_state != state) {

        qemu_input_update_buttons(dcl->con, bmap, prev_state, state);

        prev_state = state;

    }



    if (qemu_input_is_absolute()) {

        qemu_input_queue_abs(dcl->con, INPUT_AXIS_X, x,

                             real_screen->w);

        qemu_input_queue_abs(dcl->con, INPUT_AXIS_Y, y,

                             real_screen->h);

    } else {

        if (guest_cursor) {

            x -= guest_x;

            y -= guest_y;

            guest_x += x;

            guest_y += y;

            dx = x;

            dy = y;

        }

        qemu_input_queue_rel(dcl->con, INPUT_AXIS_X, dx);

        qemu_input_queue_rel(dcl->con, INPUT_AXIS_Y, dy);

    }

    qemu_input_event_sync();

}
