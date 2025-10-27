static void spapr_phb_class_init(ObjectClass *klass, void *data)

{

    PCIHostBridgeClass *hc = PCI_HOST_BRIDGE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);

    HotplugHandlerClass *hp = HOTPLUG_HANDLER_CLASS(klass);



    hc->root_bus_path = spapr_phb_root_bus_path;

    dc->realize = spapr_phb_realize;

    dc->props = spapr_phb_properties;

    dc->reset = spapr_phb_reset;

    dc->vmsd = &vmstate_spapr_pci;



    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);

    hp->plug = spapr_phb_hot_plug_child;

    hp->unplug = spapr_phb_hot_unplug_child;

}