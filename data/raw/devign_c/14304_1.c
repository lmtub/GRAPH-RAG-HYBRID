PCIBus *i440fx_init(PCII440FXState **pi440fx_state,

                    int *piix3_devfn,

                    ISABus **isa_bus, qemu_irq *pic,

                    MemoryRegion *address_space_mem,

                    MemoryRegion *address_space_io,

                    ram_addr_t ram_size,

                    hwaddr pci_hole_start,

                    hwaddr pci_hole_size,

                    hwaddr pci_hole64_start,

                    hwaddr pci_hole64_size,

                    MemoryRegion *pci_address_space,

                    MemoryRegion *ram_memory)

{

    DeviceState *dev;

    PCIBus *b;

    PCIDevice *d;

    PCIHostState *s;

    PIIX3State *piix3;

    PCII440FXState *f;

    unsigned i;



    dev = qdev_create(NULL, TYPE_I440FX_PCI_HOST_BRIDGE);

    s = PCI_HOST_BRIDGE(dev);

    b = pci_bus_new(dev, NULL, pci_address_space,

                    address_space_io, 0, TYPE_PCI_BUS);

    s->bus = b;

    object_property_add_child(qdev_get_machine(), "i440fx", OBJECT(dev), NULL);

    qdev_init_nofail(dev);



    d = pci_create_simple(b, 0, TYPE_I440FX_PCI_DEVICE);

    *pi440fx_state = I440FX_PCI_DEVICE(d);

    f = *pi440fx_state;

    f->system_memory = address_space_mem;

    f->pci_address_space = pci_address_space;

    f->ram_memory = ram_memory;

    memory_region_init_alias(&f->pci_hole, OBJECT(d), "pci-hole", f->pci_address_space,

                             pci_hole_start, pci_hole_size);

    memory_region_add_subregion(f->system_memory, pci_hole_start, &f->pci_hole);

    memory_region_init_alias(&f->pci_hole_64bit, OBJECT(d), "pci-hole64",

                             f->pci_address_space,

                             pci_hole64_start, pci_hole64_size);

    if (pci_hole64_size) {

        memory_region_add_subregion(f->system_memory, pci_hole64_start,

                                    &f->pci_hole_64bit);

    }

    memory_region_init_alias(&f->smram_region, OBJECT(d), "smram-region",

                             f->pci_address_space, 0xa0000, 0x20000);

    memory_region_add_subregion_overlap(f->system_memory, 0xa0000,

                                        &f->smram_region, 1);

    memory_region_set_enabled(&f->smram_region, false);

    init_pam(dev, f->ram_memory, f->system_memory, f->pci_address_space,

             &f->pam_regions[0], PAM_BIOS_BASE, PAM_BIOS_SIZE);

    for (i = 0; i < 12; ++i) {

        init_pam(dev, f->ram_memory, f->system_memory, f->pci_address_space,

                 &f->pam_regions[i+1], PAM_EXPAN_BASE + i * PAM_EXPAN_SIZE,

                 PAM_EXPAN_SIZE);

    }



    /* Xen supports additional interrupt routes from the PCI devices to

     * the IOAPIC: the four pins of each PCI device on the bus are also

     * connected to the IOAPIC directly.

     * These additional routes can be discovered through ACPI. */

    if (xen_enabled()) {

        piix3 = DO_UPCAST(PIIX3State, dev,

                pci_create_simple_multifunction(b, -1, true, "PIIX3-xen"));

        pci_bus_irqs(b, xen_piix3_set_irq, xen_pci_slot_get_pirq,

                piix3, XEN_PIIX_NUM_PIRQS);

    } else {

        piix3 = DO_UPCAST(PIIX3State, dev,

                pci_create_simple_multifunction(b, -1, true, "PIIX3"));

        pci_bus_irqs(b, piix3_set_irq, pci_slot_get_pirq, piix3,

                PIIX_NUM_PIRQS);

        pci_bus_set_route_irq_fn(b, piix3_route_intx_pin_to_irq);

    }

    piix3->pic = pic;

    *isa_bus = ISA_BUS(qdev_get_child_bus(DEVICE(piix3), "isa.0"));



    *piix3_devfn = piix3->dev.devfn;



    ram_size = ram_size / 8 / 1024 / 1024;

    if (ram_size > 255) {

        ram_size = 255;

    }

    d->config[0x57] = ram_size;



    i440fx_update_memory_mappings(f);



    return b;

}
