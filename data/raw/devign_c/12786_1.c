static void spapr_nvram_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VIOsPAPRDeviceClass *k = VIO_SPAPR_DEVICE_CLASS(klass);



    k->realize = spapr_nvram_realize;

    k->devnode = spapr_nvram_devnode;

    k->dt_name = "nvram";

    k->dt_type = "nvram";

    k->dt_compatible = "qemu,spapr-nvram";

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    dc->props = spapr_nvram_properties;

    dc->vmsd = &vmstate_spapr_nvram;



}