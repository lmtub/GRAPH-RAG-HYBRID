static void do_key_event(VncState *vs, int down, int keycode, int sym)
{
    /* QEMU console switch */
    switch(keycode) {
    case 0x2a:                          /* Left Shift */
    case 0x36:                          /* Right Shift */
    case 0x1d:                          /* Left CTRL */
    case 0x9d:                          /* Right CTRL */
    case 0x38:                          /* Left ALT */
    case 0xb8:                          /* Right ALT */
        if (down)
            vs->modifiers_state[keycode] = 1;
        else
            vs->modifiers_state[keycode] = 0;
        break;
    case 0x02 ... 0x0a: /* '1' to '9' keys */
        if (down && vs->modifiers_state[0x1d] && vs->modifiers_state[0x38]) {
            /* Reset the modifiers sent to the current console */
            reset_keys(vs);
            console_select(keycode - 0x02);
            return;
        break;
    case 0x3a:                        /* CapsLock */
    case 0x45:                        /* NumLock */
        if (!down)
            vs->modifiers_state[keycode] ^= 1;
        break;
    if (keycode_is_keypad(vs->vd->kbd_layout, keycode)) {
        /* If the numlock state needs to change then simulate an additional
           toggles numlock away from the VNC window.
        if (keysym_is_numlock(vs->vd->kbd_layout, sym & 0xFFFF)) {
            if (!vs->modifiers_state[0x45]) {
                vs->modifiers_state[0x45] = 1;
                press_key(vs, 0xff7f);
            if (vs->modifiers_state[0x45]) {
                vs->modifiers_state[0x45] = 0;
                press_key(vs, 0xff7f);
    if (is_graphic_console()) {
        if (keycode & 0x80)
            kbd_put_keycode(0xe0);
        if (down)
            kbd_put_keycode(keycode & 0x7f);
        else
            kbd_put_keycode(keycode | 0x80);
        /* QEMU console emulation */
        if (down) {
            int numlock = vs->modifiers_state[0x45];
            switch (keycode) {
            case 0x2a:                          /* Left Shift */
            case 0x36:                          /* Right Shift */
            case 0x1d:                          /* Left CTRL */
            case 0x9d:                          /* Right CTRL */
            case 0x38:                          /* Left ALT */
            case 0xb8:                          /* Right ALT */
                break;
            case 0xc8:
                kbd_put_keysym(QEMU_KEY_UP);
                break;
            case 0xd0:
                kbd_put_keysym(QEMU_KEY_DOWN);
                break;
            case 0xcb:
                kbd_put_keysym(QEMU_KEY_LEFT);
                break;
            case 0xcd:
                kbd_put_keysym(QEMU_KEY_RIGHT);
                break;
            case 0xd3:
                kbd_put_keysym(QEMU_KEY_DELETE);
                break;
            case 0xc7:
                kbd_put_keysym(QEMU_KEY_HOME);
                break;
            case 0xcf:
                kbd_put_keysym(QEMU_KEY_END);
                break;
            case 0xc9:
                kbd_put_keysym(QEMU_KEY_PAGEUP);
                break;
            case 0xd1:
                kbd_put_keysym(QEMU_KEY_PAGEDOWN);
                break;
            case 0x47:
                kbd_put_keysym(numlock ? '7' : QEMU_KEY_HOME);
                break;
            case 0x48:
                kbd_put_keysym(numlock ? '8' : QEMU_KEY_UP);
                break;
            case 0x49:
                kbd_put_keysym(numlock ? '9' : QEMU_KEY_PAGEUP);
                break;
            case 0x4b:
                kbd_put_keysym(numlock ? '4' : QEMU_KEY_LEFT);
                break;
            case 0x4c:
                kbd_put_keysym('5');
                break;
            case 0x4d:
                kbd_put_keysym(numlock ? '6' : QEMU_KEY_RIGHT);
                break;
            case 0x4f:
                kbd_put_keysym(numlock ? '1' : QEMU_KEY_END);
                break;
            case 0x50:
                kbd_put_keysym(numlock ? '2' : QEMU_KEY_DOWN);
                break;
            case 0x51:
                kbd_put_keysym(numlock ? '3' : QEMU_KEY_PAGEDOWN);
                break;
            case 0x52:
                kbd_put_keysym('0');
                break;
            case 0x53:
                kbd_put_keysym(numlock ? '.' : QEMU_KEY_DELETE);
                break;
            case 0xb5:
                kbd_put_keysym('/');
                break;
            case 0x37:
                kbd_put_keysym('*');
                break;
            case 0x4a:
                kbd_put_keysym('-');
                break;
            case 0x4e:
                kbd_put_keysym('+');
                break;
            case 0x9c:
                kbd_put_keysym('\n');
                break;
            default:
                kbd_put_keysym(sym);
                break;