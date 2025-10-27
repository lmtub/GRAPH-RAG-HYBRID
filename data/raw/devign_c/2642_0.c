void kbd_put_keycode(int keycode)

{

    if (!runstate_is_running()) {

        return;

    }

    if (qemu_put_kbd_event) {

        qemu_put_kbd_event(qemu_put_kbd_event_opaque, keycode);

    }

}
