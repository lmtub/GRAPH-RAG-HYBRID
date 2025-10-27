static void ps2_reset_keyboard(PS2KbdState *s)

{

    trace_ps2_reset_keyboard(s);

    s->scan_enabled = 1;

    s->scancode_set = 2;


    ps2_set_ledstate(s, 0);

}