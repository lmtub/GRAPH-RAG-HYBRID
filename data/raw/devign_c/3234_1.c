static void adb_kbd_put_keycode(void *opaque, int keycode)

{

    KBDState *s = opaque;



    if (s->count < sizeof(s->data)) {

        s->data[s->wptr] = keycode;

        if (++s->wptr == sizeof(s->data))

            s->wptr = 0;

        s->count++;

    }

}
