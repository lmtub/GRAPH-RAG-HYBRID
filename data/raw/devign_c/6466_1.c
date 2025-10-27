static void adb_kbd_realizefn(DeviceState *dev, Error **errp)

{

    ADBKeyboardClass *akc = ADB_KEYBOARD_GET_CLASS(dev);

    akc->parent_realize(dev, errp);

    qemu_input_handler_register(dev, &adb_keyboard_handler);

}
