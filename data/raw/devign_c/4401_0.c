void qemu_input_event_send_key_number(QemuConsole *src, int num, bool down)

{

    KeyValue *key = g_new0(KeyValue, 1);

    key->type = KEY_VALUE_KIND_NUMBER;

    key->u.number = num;

    qemu_input_event_send_key(src, key, down);

}
