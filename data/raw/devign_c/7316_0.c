static void legacy_kbd_event(DeviceState *dev, QemuConsole *src,

                             InputEvent *evt)

{

    QEMUPutKbdEntry *entry = (QEMUPutKbdEntry *)dev;

    int scancodes[3], i, count;



    if (!entry || !entry->put_kbd) {

        return;

    }

    count = qemu_input_key_value_to_scancode(evt->key->key,

                                             evt->key->down,

                                             scancodes);

    for (i = 0; i < count; i++) {

        entry->put_kbd(entry->opaque, scancodes[i]);

    }

}
