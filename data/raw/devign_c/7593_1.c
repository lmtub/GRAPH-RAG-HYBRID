static void pxb_host_class_init(ObjectClass *class, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(class);

    SysBusDeviceClass *sbc = SYS_BUS_DEVICE_CLASS(class);

    PCIHostBridgeClass *hc = PCI_HOST_BRIDGE_CLASS(class);



    dc->fw_name = "pci";



    sbc->explicit_ofw_unit_address = pxb_host_ofw_unit_address;

    hc->root_bus_path = pxb_host_root_bus_path;

}