static void vmxnet3_deactivate_device(VMXNET3State *s)

{

    VMW_CBPRN("Deactivating vmxnet3...");

    s->device_active = false;

}
