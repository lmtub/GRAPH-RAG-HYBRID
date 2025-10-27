static void adb_mouse_initfn(Object *obj)

{

    ADBDevice *d = ADB_DEVICE(obj);



    d->devaddr = ADB_DEVID_MOUSE;

}
