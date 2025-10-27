static void vmgenid_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->vmsd = &vmstate_vmgenid;

    dc->realize = vmgenid_realize;

    dc->hotpluggable = false;

    dc->props = vmgenid_properties;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);



    object_class_property_add_str(klass, VMGENID_GUID, NULL,

                                  vmgenid_set_guid, NULL);

    object_class_property_set_description(klass, VMGENID_GUID,

                                    "Set Global Unique Identifier "

                                    "(big-endian) or auto for random value",

                                    NULL);

}
