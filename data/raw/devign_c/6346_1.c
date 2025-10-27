static void q35_host_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIHostBridgeClass *hc = PCI_HOST_BRIDGE_CLASS(klass);



    hc->root_bus_path = q35_host_root_bus_path;

    dc->realize = q35_host_realize;

    dc->props = mch_props;

    /* Reason: needs to be wired up by pc_q35_init */

    dc->user_creatable = false;

    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);

    dc->fw_name = "pci";

}
