static void i440fx_pcihost_initfn(Object *obj)

{

    PCIHostState *s = PCI_HOST_BRIDGE(obj);

    I440FXState *d = I440FX_PCI_HOST_BRIDGE(obj);



    memory_region_init_io(&s->conf_mem, obj, &pci_host_conf_le_ops, s,

                          "pci-conf-idx", 4);

    memory_region_init_io(&s->data_mem, obj, &pci_host_data_le_ops, s,

                          "pci-conf-data", 4);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE_START, "int",

                        i440fx_pcihost_get_pci_hole_start,

                        NULL, NULL, NULL, NULL);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE_END, "int",

                        i440fx_pcihost_get_pci_hole_end,

                        NULL, NULL, NULL, NULL);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE64_START, "int",

                        i440fx_pcihost_get_pci_hole64_start,

                        NULL, NULL, NULL, NULL);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE64_END, "int",

                        i440fx_pcihost_get_pci_hole64_end,

                        NULL, NULL, NULL, NULL);



    d->pci_info.w32.end = IO_APIC_DEFAULT_ADDRESS;

}
