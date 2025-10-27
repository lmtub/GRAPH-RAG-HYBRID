int qemu_input_key_value_to_scancode(const KeyValue *value, bool down,

                                     int *codes)

{

    int keycode = qemu_input_key_value_to_number(value);

    int count = 0;



    if (value->type == KEY_VALUE_KIND_QCODE &&

        value->u.qcode == Q_KEY_CODE_PAUSE) {

        /* specific case */

        int v = down ? 0 : 0x80;

        codes[count++] = 0xe1;

        codes[count++] = 0x1d | v;

        codes[count++] = 0x45 | v;

        return count;

    }

    if (keycode & SCANCODE_GREY) {

        codes[count++] = SCANCODE_EMUL0;

        keycode &= ~SCANCODE_GREY;

    }

    if (!down) {

        keycode |= SCANCODE_UP;

    }

    codes[count++] = keycode;



    return count;

}
