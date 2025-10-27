static void i440fx_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->no_hotplug = 1;

    k->init = i440fx_initfn;

    k->config_write = i440fx_write_config;

    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_82441;

    k->revision = 0x02;

    k->class_id = PCI_CLASS_BRIDGE_HOST;

    dc->desc = "Host bridge";

    dc->no_user = 1;

    dc->vmsd = &vmstate_i440fx;

}
