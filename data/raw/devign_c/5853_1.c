static void vfio_calxeda_xgmac_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VFIOCalxedaXgmacDeviceClass *vcxc =

        VFIO_CALXEDA_XGMAC_DEVICE_CLASS(klass);

    vcxc->parent_realize = dc->realize;

    dc->realize = calxeda_xgmac_realize;

    dc->desc = "VFIO Calxeda XGMAC";

    dc->vmsd = &vfio_platform_calxeda_xgmac_vmstate;



}