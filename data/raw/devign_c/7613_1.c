void stellaris_gamepad_init(int n, qemu_irq *irq, const int *keycode)

{

    gamepad_state *s;

    int i;



    s = (gamepad_state *)g_malloc0(sizeof (gamepad_state));

    s->buttons = (gamepad_button *)g_malloc0(n * sizeof (gamepad_button));

    for (i = 0; i < n; i++) {

        s->buttons[i].irq = irq[i];

        s->buttons[i].keycode = keycode[i];

    }

    s->num_buttons = n;

    qemu_add_kbd_event_handler(stellaris_gamepad_put_key, s);

    vmstate_register(NULL, -1, &vmstate_stellaris_gamepad, s);

}
