static void acpi_dsdt_add_pci(Aml *scope, const MemMapEntry *memmap,

                              uint32_t irq, bool use_highmem)

{

    Aml *method, *crs, *ifctx, *UUID, *ifctx1, *elsectx, *buf;

    int i, bus_no;

    hwaddr base_mmio = memmap[VIRT_PCIE_MMIO].base;

    hwaddr size_mmio = memmap[VIRT_PCIE_MMIO].size;

    hwaddr base_pio = memmap[VIRT_PCIE_PIO].base;

    hwaddr size_pio = memmap[VIRT_PCIE_PIO].size;

    hwaddr base_ecam = memmap[VIRT_PCIE_ECAM].base;

    hwaddr size_ecam = memmap[VIRT_PCIE_ECAM].size;

    int nr_pcie_buses = size_ecam / PCIE_MMCFG_SIZE_MIN;



    Aml *dev = aml_device("%s", "PCI0");

    aml_append(dev, aml_name_decl("_HID", aml_string("PNP0A08")));

    aml_append(dev, aml_name_decl("_CID", aml_string("PNP0A03")));

    aml_append(dev, aml_name_decl("_SEG", aml_int(0)));

    aml_append(dev, aml_name_decl("_BBN", aml_int(0)));

    aml_append(dev, aml_name_decl("_ADR", aml_int(0)));

    aml_append(dev, aml_name_decl("_UID", aml_string("PCI0")));

    aml_append(dev, aml_name_decl("_STR", aml_unicode("PCIe 0 Device")));

    aml_append(dev, aml_name_decl("_CCA", aml_int(1)));



    /* Declare the PCI Routing Table. */

    Aml *rt_pkg = aml_package(nr_pcie_buses * PCI_NUM_PINS);

    for (bus_no = 0; bus_no < nr_pcie_buses; bus_no++) {

        for (i = 0; i < PCI_NUM_PINS; i++) {

            int gsi = (i + bus_no) % PCI_NUM_PINS;

            Aml *pkg = aml_package(4);

            aml_append(pkg, aml_int((bus_no << 16) | 0xFFFF));

            aml_append(pkg, aml_int(i));

            aml_append(pkg, aml_name("GSI%d", gsi));

            aml_append(pkg, aml_int(0));

            aml_append(rt_pkg, pkg);

        }

    }

    aml_append(dev, aml_name_decl("_PRT", rt_pkg));



    /* Create GSI link device */

    for (i = 0; i < PCI_NUM_PINS; i++) {

        uint32_t irqs =  irq + i;

        Aml *dev_gsi = aml_device("GSI%d", i);

        aml_append(dev_gsi, aml_name_decl("_HID", aml_string("PNP0C0F")));

        aml_append(dev_gsi, aml_name_decl("_UID", aml_int(0)));

        crs = aml_resource_template();

        aml_append(crs,

                   aml_interrupt(AML_CONSUMER, AML_LEVEL, AML_ACTIVE_HIGH,

                                 AML_EXCLUSIVE, &irqs, 1));

        aml_append(dev_gsi, aml_name_decl("_PRS", crs));

        crs = aml_resource_template();

        aml_append(crs,

                   aml_interrupt(AML_CONSUMER, AML_LEVEL, AML_ACTIVE_HIGH,

                                 AML_EXCLUSIVE, &irqs, 1));

        aml_append(dev_gsi, aml_name_decl("_CRS", crs));

        method = aml_method("_SRS", 1, AML_NOTSERIALIZED);

        aml_append(dev_gsi, method);

        aml_append(dev, dev_gsi);

    }



    method = aml_method("_CBA", 0, AML_NOTSERIALIZED);

    aml_append(method, aml_return(aml_int(base_ecam)));

    aml_append(dev, method);



    method = aml_method("_CRS", 0, AML_NOTSERIALIZED);

    Aml *rbuf = aml_resource_template();

    aml_append(rbuf,

        aml_word_bus_number(AML_MIN_FIXED, AML_MAX_FIXED, AML_POS_DECODE,

                            0x0000, 0x0000, nr_pcie_buses - 1, 0x0000,

                            nr_pcie_buses));

    aml_append(rbuf,

        aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED, AML_MAX_FIXED,

                         AML_NON_CACHEABLE, AML_READ_WRITE, 0x0000, base_mmio,

                         base_mmio + size_mmio - 1, 0x0000, size_mmio));

    aml_append(rbuf,

        aml_dword_io(AML_MIN_FIXED, AML_MAX_FIXED, AML_POS_DECODE,

                     AML_ENTIRE_RANGE, 0x0000, 0x0000, size_pio - 1, base_pio,

                     size_pio));



    if (use_highmem) {

        hwaddr base_mmio_high = memmap[VIRT_PCIE_MMIO_HIGH].base;

        hwaddr size_mmio_high = memmap[VIRT_PCIE_MMIO_HIGH].size;



        aml_append(rbuf,

            aml_qword_memory(AML_POS_DECODE, AML_MIN_FIXED, AML_MAX_FIXED,

                             AML_NON_CACHEABLE, AML_READ_WRITE, 0x0000,

                             base_mmio_high, base_mmio_high, 0x0000,

                             size_mmio_high));

    }



    aml_append(method, aml_name_decl("RBUF", rbuf));

    aml_append(method, aml_return(rbuf));

    aml_append(dev, method);



    /* Declare an _OSC (OS Control Handoff) method */

    aml_append(dev, aml_name_decl("SUPP", aml_int(0)));

    aml_append(dev, aml_name_decl("CTRL", aml_int(0)));

    method = aml_method("_OSC", 4, AML_NOTSERIALIZED);

    aml_append(method,

        aml_create_dword_field(aml_arg(3), aml_int(0), "CDW1"));



    /* PCI Firmware Specification 3.0

     * 4.5.1. _OSC Interface for PCI Host Bridge Devices

     * The _OSC interface for a PCI/PCI-X/PCI Express hierarchy is

     * identified by the Universal Unique IDentifier (UUID)

     * 33DB4D5B-1FF7-401C-9657-7441C03DD766

     */

    UUID = aml_touuid("33DB4D5B-1FF7-401C-9657-7441C03DD766");

    ifctx = aml_if(aml_equal(aml_arg(0), UUID));

    aml_append(ifctx,

        aml_create_dword_field(aml_arg(3), aml_int(4), "CDW2"));

    aml_append(ifctx,

        aml_create_dword_field(aml_arg(3), aml_int(8), "CDW3"));

    aml_append(ifctx, aml_store(aml_name("CDW2"), aml_name("SUPP")));

    aml_append(ifctx, aml_store(aml_name("CDW3"), aml_name("CTRL")));

    aml_append(ifctx, aml_store(aml_and(aml_name("CTRL"), aml_int(0x1D), NULL),

                                aml_name("CTRL")));



    ifctx1 = aml_if(aml_lnot(aml_equal(aml_arg(1), aml_int(0x1))));

    aml_append(ifctx1, aml_store(aml_or(aml_name("CDW1"), aml_int(0x08), NULL),

                                 aml_name("CDW1")));

    aml_append(ifctx, ifctx1);



    ifctx1 = aml_if(aml_lnot(aml_equal(aml_name("CDW3"), aml_name("CTRL"))));

    aml_append(ifctx1, aml_store(aml_or(aml_name("CDW1"), aml_int(0x10), NULL),

                                 aml_name("CDW1")));

    aml_append(ifctx, ifctx1);



    aml_append(ifctx, aml_store(aml_name("CTRL"), aml_name("CDW3")));

    aml_append(ifctx, aml_return(aml_arg(3)));

    aml_append(method, ifctx);



    elsectx = aml_else();

    aml_append(elsectx, aml_store(aml_or(aml_name("CDW1"), aml_int(4), NULL),

                                  aml_name("CDW1")));

    aml_append(elsectx, aml_return(aml_arg(3)));

    aml_append(method, elsectx);

    aml_append(dev, method);



    method = aml_method("_DSM", 4, AML_NOTSERIALIZED);



    /* PCI Firmware Specification 3.0

     * 4.6.1. _DSM for PCI Express Slot Information

     * The UUID in _DSM in this context is

     * {E5C937D0-3553-4D7A-9117-EA4D19C3434D}

     */

    UUID = aml_touuid("E5C937D0-3553-4D7A-9117-EA4D19C3434D");

    ifctx = aml_if(aml_equal(aml_arg(0), UUID));

    ifctx1 = aml_if(aml_equal(aml_arg(2), aml_int(0)));

    uint8_t byte_list[1] = {1};

    buf = aml_buffer(1, byte_list);

    aml_append(ifctx1, aml_return(buf));

    aml_append(ifctx, ifctx1);

    aml_append(method, ifctx);



    byte_list[0] = 0;

    buf = aml_buffer(1, byte_list);

    aml_append(method, aml_return(buf));

    aml_append(dev, method);



    Aml *dev_rp0 = aml_device("%s", "RP0");

    aml_append(dev_rp0, aml_name_decl("_ADR", aml_int(0)));

    aml_append(dev, dev_rp0);

    aml_append(scope, dev);

}
