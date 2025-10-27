static void sysbus_esp_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = sysbus_esp_realize;

    dc->reset = sysbus_esp_hard_reset;

    dc->vmsd = &vmstate_sysbus_esp_scsi;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);






}