static void adb_mouse_reset(DeviceState *dev)

{

    ADBDevice *d = ADB_DEVICE(dev);

    MouseState *s = ADB_MOUSE(dev);



    d->handler = 2;

    d->devaddr = ADB_DEVID_MOUSE;

    s->last_buttons_state = s->buttons_state = 0;

    s->dx = s->dy = s->dz = 0;

}
