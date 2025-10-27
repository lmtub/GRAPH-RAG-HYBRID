static void ich9_lpc_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);

    dc->reset = ich9_lpc_reset;

    k->init = ich9_lpc_initfn;

    dc->vmsd = &vmstate_ich9_lpc;

    dc->no_user = 1;

    k->config_write = ich9_lpc_config_write;

    dc->desc = "ICH9 LPC bridge";

    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_ICH9_8;

    k->revision = ICH9_A2_LPC_REVISION;

    k->class_id = PCI_CLASS_BRIDGE_ISA;



}
