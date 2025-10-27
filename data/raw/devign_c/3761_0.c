void replay_save_input_event(InputEvent *evt)

{

    InputKeyEvent *key;

    InputBtnEvent *btn;

    InputMoveEvent *move;

    replay_put_dword(evt->type);



    switch (evt->type) {

    case INPUT_EVENT_KIND_KEY:

        key = evt->u.key;

        replay_put_dword(key->key->type);



        switch (key->key->type) {

        case KEY_VALUE_KIND_NUMBER:

            replay_put_qword(key->key->u.number);

            replay_put_byte(key->down);

            break;

        case KEY_VALUE_KIND_QCODE:

            replay_put_dword(key->key->u.qcode);

            replay_put_byte(key->down);

            break;

        case KEY_VALUE_KIND__MAX:

            /* keep gcc happy */

            break;

        }

        break;

    case INPUT_EVENT_KIND_BTN:

        btn = evt->u.btn;

        replay_put_dword(btn->button);

        replay_put_byte(btn->down);

        break;

    case INPUT_EVENT_KIND_REL:

        move = evt->u.rel;

        replay_put_dword(move->axis);

        replay_put_qword(move->value);

        break;

    case INPUT_EVENT_KIND_ABS:

        move = evt->u.abs;

        replay_put_dword(move->axis);

        replay_put_qword(move->value);

        break;

    case INPUT_EVENT_KIND__MAX:

        /* keep gcc happy */

        break;

    }

}
