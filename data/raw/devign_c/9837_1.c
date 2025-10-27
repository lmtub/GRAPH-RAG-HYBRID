static void adb_kbd_initfn(Object *obj)

{

    ADBDevice *d = ADB_DEVICE(obj);



    d->devaddr = ADB_DEVID_KEYBOARD;

}
