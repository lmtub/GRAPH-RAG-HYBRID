static void spice_update_buttons(QemuSpicePointer *pointer,

                                 int wheel, uint32_t button_mask)

{

    static uint32_t bmap[INPUT_BUTTON__MAX] = {

        [INPUT_BUTTON_LEFT]        = 0x01,

        [INPUT_BUTTON_MIDDLE]      = 0x04,

        [INPUT_BUTTON_RIGHT]       = 0x02,

        [INPUT_BUTTON_WHEEL_UP]    = 0x10,

        [INPUT_BUTTON_WHEEL_DOWN]  = 0x20,

    };



    if (wheel < 0) {

        button_mask |= 0x10;

    }

    if (wheel > 0) {

        button_mask |= 0x20;

    }



    if (pointer->last_bmask == button_mask) {

        return;

    }

    qemu_input_update_buttons(NULL, bmap, pointer->last_bmask, button_mask);

    pointer->last_bmask = button_mask;

}
