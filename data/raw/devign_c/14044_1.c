static void adb_mouse_realizefn(DeviceState *dev, Error **errp)

{

    MouseState *s = ADB_MOUSE(dev);

    ADBMouseClass *amc = ADB_MOUSE_GET_CLASS(dev);



    amc->parent_realize(dev, errp);



    qemu_add_mouse_event_handler(adb_mouse_event, s, 0, "QEMU ADB Mouse");

}
