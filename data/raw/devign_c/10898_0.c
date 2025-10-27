static void q35_host_initfn(Object *obj)

{

    Q35PCIHost *s = Q35_HOST_DEVICE(obj);

    PCIHostState *phb = PCI_HOST_BRIDGE(obj);



    memory_region_init_io(&phb->conf_mem, obj, &pci_host_conf_le_ops, phb,

                          "pci-conf-idx", 4);

    memory_region_init_io(&phb->data_mem, obj, &pci_host_data_le_ops, phb,

                          "pci-conf-data", 4);



    object_initialize(&s->mch, sizeof(s->mch), TYPE_MCH_PCI_DEVICE);

    object_property_add_child(OBJECT(s), "mch", OBJECT(&s->mch), NULL);

    qdev_prop_set_uint32(DEVICE(&s->mch), "addr", PCI_DEVFN(0, 0));

    qdev_prop_set_bit(DEVICE(&s->mch), "multifunction", false);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE_START, "int",

                        q35_host_get_pci_hole_start,

                        NULL, NULL, NULL, NULL);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE_END, "int",

                        q35_host_get_pci_hole_end,

                        NULL, NULL, NULL, NULL);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE64_START, "int",

                        q35_host_get_pci_hole64_start,

                        NULL, NULL, NULL, NULL);



    object_property_add(obj, PCI_HOST_PROP_PCI_HOLE64_END, "int",

                        q35_host_get_pci_hole64_end,

                        NULL, NULL, NULL, NULL);



    object_property_add(obj, PCIE_HOST_MCFG_SIZE, "int",

                        q35_host_get_mmcfg_size,

                        NULL, NULL, NULL, NULL);



    object_property_add_link(obj, MCH_HOST_PROP_RAM_MEM, TYPE_MEMORY_REGION,

                             (Object **) &s->mch.ram_memory,

                             qdev_prop_allow_set_link_before_realize, 0, NULL);



    object_property_add_link(obj, MCH_HOST_PROP_PCI_MEM, TYPE_MEMORY_REGION,

                             (Object **) &s->mch.pci_address_space,

                             qdev_prop_allow_set_link_before_realize, 0, NULL);



    object_property_add_link(obj, MCH_HOST_PROP_SYSTEM_MEM, TYPE_MEMORY_REGION,

                             (Object **) &s->mch.system_memory,

                             qdev_prop_allow_set_link_before_realize, 0, NULL);



    object_property_add_link(obj, MCH_HOST_PROP_IO_MEM, TYPE_MEMORY_REGION,

                             (Object **) &s->mch.address_space_io,

                             qdev_prop_allow_set_link_before_realize, 0, NULL);



    /* Leave enough space for the biggest MCFG BAR */

    /* TODO: this matches current bios behaviour, but

     * it's not a power of two, which means an MTRR

     * can't cover it exactly.

     */

    s->mch.pci_hole.begin = MCH_HOST_BRIDGE_PCIEXBAR_DEFAULT +

        MCH_HOST_BRIDGE_PCIEXBAR_MAX;

    s->mch.pci_hole.end = IO_APIC_DEFAULT_ADDRESS;

}
