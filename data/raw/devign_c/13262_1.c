static void i440fx_pcihost_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIHostBridgeClass *hc = PCI_HOST_BRIDGE_CLASS(klass);



    hc->root_bus_path = i440fx_pcihost_root_bus_path;

    dc->realize = i440fx_pcihost_realize;

    dc->fw_name = "pci";

    dc->props = i440fx_props;



}