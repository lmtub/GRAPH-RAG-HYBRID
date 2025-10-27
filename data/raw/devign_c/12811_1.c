static void ich9_smb_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_ICH9_6;

    k->revision = ICH9_A2_SMB_REVISION;

    k->class_id = PCI_CLASS_SERIAL_SMBUS;

    dc->no_user = 1;

    dc->vmsd = &vmstate_ich9_smbus;

    dc->desc = "ICH9 SMBUS Bridge";

    k->init = ich9_smbus_initfn;

    k->config_write = ich9_smbus_write_config;

}
