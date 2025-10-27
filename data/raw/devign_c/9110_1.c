static void vmgenid_realize(DeviceState *dev, Error **errp)

{

    VmGenIdState *vms = VMGENID(dev);



    if (!vms->write_pointer_available) {

        error_setg(errp, "%s requires DMA write support in fw_cfg, "

                   "which this machine type does not provide", VMGENID_DEVICE);

        return;

    }



    /* Given that this function is executing, there is at least one VMGENID

     * device. Check if there are several.

     */

    if (!find_vmgenid_dev()) {

        error_setg(errp, "at most one %s device is permitted", VMGENID_DEVICE);

        return;

    }



    qemu_register_reset(vmgenid_handle_reset, vms);

}
