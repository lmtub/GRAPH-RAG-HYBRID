InputEvent *replay_read_input_event(void)

{

    InputEvent evt;

    KeyValue keyValue;

    InputKeyEvent key;

    key.key = &keyValue;

    InputBtnEvent btn;

    InputMoveEvent rel;

    InputMoveEvent abs;



    evt.type = replay_get_dword();

    switch (evt.type) {

    case INPUT_EVENT_KIND_KEY:

        evt.u.key = &key;

        evt.u.key->key->type = replay_get_dword();



        switch (evt.u.key->key->type) {

        case KEY_VALUE_KIND_NUMBER:

            evt.u.key->key->u.number = replay_get_qword();

            evt.u.key->down = replay_get_byte();

            break;

        case KEY_VALUE_KIND_QCODE:

            evt.u.key->key->u.qcode = (QKeyCode)replay_get_dword();

            evt.u.key->down = replay_get_byte();

            break;

        case KEY_VALUE_KIND__MAX:

            /* keep gcc happy */

            break;

        }

        break;

    case INPUT_EVENT_KIND_BTN:

        evt.u.btn = &btn;

        evt.u.btn->button = (InputButton)replay_get_dword();

        evt.u.btn->down = replay_get_byte();

        break;

    case INPUT_EVENT_KIND_REL:

        evt.u.rel = &rel;

        evt.u.rel->axis = (InputAxis)replay_get_dword();

        evt.u.rel->value = replay_get_qword();

        break;

    case INPUT_EVENT_KIND_ABS:

        evt.u.abs = &abs;

        evt.u.abs->axis = (InputAxis)replay_get_dword();

        evt.u.abs->value = replay_get_qword();

        break;

    case INPUT_EVENT_KIND__MAX:

        /* keep gcc happy */

        break;

    }



    return qapi_clone_InputEvent(&evt);

}
