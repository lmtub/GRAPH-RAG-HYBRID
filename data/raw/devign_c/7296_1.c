static void vfio_amd_xgbe_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VFIOAmdXgbeDeviceClass *vcxc =

        VFIO_AMD_XGBE_DEVICE_CLASS(klass);

    vcxc->parent_realize = dc->realize;

    dc->realize = amd_xgbe_realize;

    dc->desc = "VFIO AMD XGBE";

    dc->vmsd = &vfio_platform_amd_xgbe_vmstate;



}