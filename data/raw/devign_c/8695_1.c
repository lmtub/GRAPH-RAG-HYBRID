static int adb_mouse_poll(ADBDevice *d, uint8_t *obuf)

{

    MouseState *s = ADB_MOUSE(d);

    int dx, dy;



    if (s->last_buttons_state == s->buttons_state &&

        s->dx == 0 && s->dy == 0)

        return 0;



    dx = s->dx;

    if (dx < -63)

        dx = -63;

    else if (dx > 63)

        dx = 63;



    dy = s->dy;

    if (dy < -63)

        dy = -63;

    else if (dy > 63)

        dy = 63;



    s->dx -= dx;

    s->dy -= dy;

    s->last_buttons_state = s->buttons_state;



    dx &= 0x7f;

    dy &= 0x7f;



    if (!(s->buttons_state & MOUSE_EVENT_LBUTTON))

        dy |= 0x80;

    if (!(s->buttons_state & MOUSE_EVENT_RBUTTON))

        dx |= 0x80;



    obuf[0] = dy;

    obuf[1] = dx;

    return 2;

}
