static void sysbus_ahci_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = sysbus_ahci_realize;

    dc->vmsd = &vmstate_sysbus_ahci;

    dc->props = sysbus_ahci_properties;

    dc->reset = sysbus_ahci_reset;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);






}