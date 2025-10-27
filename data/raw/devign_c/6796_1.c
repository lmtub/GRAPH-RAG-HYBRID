static void adb_register_types(void)

{

    type_register_static(&adb_bus_type_info);

    type_register_static(&adb_device_type_info);

    type_register_static(&adb_kbd_type_info);

    type_register_static(&adb_mouse_type_info);

}
