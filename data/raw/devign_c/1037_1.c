build_ssdt(GArray *table_data, GArray *linker,

           AcpiCpuInfo *cpu, AcpiPmInfo *pm, AcpiMiscInfo *misc,

           PcPciInfo *pci, PcGuestInfo *guest_info)

{

    MachineState *machine = MACHINE(qdev_get_machine());

    uint32_t nr_mem = machine->ram_slots;

    unsigned acpi_cpus = guest_info->apic_id_limit;

    Aml *ssdt, *sb_scope, *scope, *pkg, *dev, *method, *crs, *field, *ifctx;

    PCIBus *bus = NULL;

    GPtrArray *io_ranges = g_ptr_array_new_with_free_func(crs_range_free);

    GPtrArray *mem_ranges = g_ptr_array_new_with_free_func(crs_range_free);

    CrsRangeEntry *entry;

    int root_bus_limit = 0xFF;

    int i;



    ssdt = init_aml_allocator();

    /* The current AML generator can cover the APIC ID range [0..255],

     * inclusive, for VCPU hotplug. */

    QEMU_BUILD_BUG_ON(ACPI_CPU_HOTPLUG_ID_LIMIT > 256);

    g_assert(acpi_cpus <= ACPI_CPU_HOTPLUG_ID_LIMIT);



    /* Reserve space for header */

    acpi_data_push(ssdt->buf, sizeof(AcpiTableHeader));



    /* Extra PCI root buses are implemented  only for i440fx */

    bus = find_i440fx();

    if (bus) {

        QLIST_FOREACH(bus, &bus->child, sibling) {

            uint8_t bus_num = pci_bus_num(bus);

            uint8_t numa_node = pci_bus_numa_node(bus);



            /* look only for expander root buses */

            if (!pci_bus_is_root(bus)) {

                continue;

            }



            if (bus_num < root_bus_limit) {

                root_bus_limit = bus_num - 1;

            }



            scope = aml_scope("\\_SB");

            dev = aml_device("PC%.02X", bus_num);

            aml_append(dev, aml_name_decl("_UID", aml_int(bus_num)));

            aml_append(dev, aml_name_decl("_HID", aml_eisaid("PNP0A03")));

            aml_append(dev, aml_name_decl("_BBN", aml_int(bus_num)));



            if (numa_node != NUMA_NODE_UNASSIGNED) {

                aml_append(dev, aml_name_decl("_PXM", aml_int(numa_node)));

            }



            aml_append(dev, build_prt());

            crs = build_crs(PCI_HOST_BRIDGE(BUS(bus)->parent),

                            io_ranges, mem_ranges);

            aml_append(dev, aml_name_decl("_CRS", crs));

            aml_append(scope, dev);

            aml_append(ssdt, scope);

        }

    }



    scope = aml_scope("\\_SB.PCI0");

    /* build PCI0._CRS */

    crs = aml_resource_template();

    aml_append(crs,

        aml_word_bus_number(AML_MIN_FIXED, AML_MAX_FIXED, AML_POS_DECODE,

                            0x0000, 0x0, root_bus_limit,

                            0x0000, root_bus_limit + 1));

    aml_append(crs, aml_io(AML_DECODE16, 0x0CF8, 0x0CF8, 0x01, 0x08));



    aml_append(crs,

        aml_word_io(AML_MIN_FIXED, AML_MAX_FIXED,

                    AML_POS_DECODE, AML_ENTIRE_RANGE,

                    0x0000, 0x0000, 0x0CF7, 0x0000, 0x0CF8));



    crs_replace_with_free_ranges(io_ranges, 0x0D00, 0xFFFF);

    for (i = 0; i < io_ranges->len; i++) {

        entry = g_ptr_array_index(io_ranges, i);

        aml_append(crs,

            aml_word_io(AML_MIN_FIXED, AML_MAX_FIXED,

                        AML_POS_DECODE, AML_ENTIRE_RANGE,

                        0x0000, entry->base, entry->limit,

                        0x0000, entry->limit - entry->base + 1));

    }



    aml_append(crs,

        aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED, AML_MAX_FIXED,

                         AML_CACHEABLE, AML_READ_WRITE,

                         0, 0x000A0000, 0x000BFFFF, 0, 0x00020000));



    crs_replace_with_free_ranges(mem_ranges, pci->w32.begin, pci->w32.end - 1);

    for (i = 0; i < mem_ranges->len; i++) {

        entry = g_ptr_array_index(mem_ranges, i);

        aml_append(crs,

            aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED, AML_MAX_FIXED,

                             AML_NON_CACHEABLE, AML_READ_WRITE,

                             0, entry->base, entry->limit,

                             0, entry->limit - entry->base + 1));

    }



    if (pci->w64.begin) {

        aml_append(crs,

            aml_qword_memory(AML_POS_DECODE, AML_MIN_FIXED, AML_MAX_FIXED,

                             AML_CACHEABLE, AML_READ_WRITE,

                             0, pci->w64.begin, pci->w64.end - 1, 0,

                             pci->w64.end - pci->w64.begin));

    }

    aml_append(scope, aml_name_decl("_CRS", crs));



    /* reserve GPE0 block resources */

    dev = aml_device("GPE0");

    aml_append(dev, aml_name_decl("_HID", aml_string("PNP0A06")));

    aml_append(dev, aml_name_decl("_UID", aml_string("GPE0 resources")));

    /* device present, functioning, decoding, not shown in UI */

    aml_append(dev, aml_name_decl("_STA", aml_int(0xB)));

    crs = aml_resource_template();

    aml_append(crs,

        aml_io(AML_DECODE16, pm->gpe0_blk, pm->gpe0_blk, 1, pm->gpe0_blk_len)

    );

    aml_append(dev, aml_name_decl("_CRS", crs));

    aml_append(scope, dev);



    g_ptr_array_free(io_ranges, true);

    g_ptr_array_free(mem_ranges, true);



    /* reserve PCIHP resources */

    if (pm->pcihp_io_len) {

        dev = aml_device("PHPR");

        aml_append(dev, aml_name_decl("_HID", aml_string("PNP0A06")));

        aml_append(dev,

            aml_name_decl("_UID", aml_string("PCI Hotplug resources")));

        /* device present, functioning, decoding, not shown in UI */

        aml_append(dev, aml_name_decl("_STA", aml_int(0xB)));

        crs = aml_resource_template();

        aml_append(crs,

            aml_io(AML_DECODE16, pm->pcihp_io_base, pm->pcihp_io_base, 1,

                   pm->pcihp_io_len)

        );

        aml_append(dev, aml_name_decl("_CRS", crs));

        aml_append(scope, dev);

    }

    aml_append(ssdt, scope);



    /*  create S3_ / S4_ / S5_ packages if necessary */

    scope = aml_scope("\\");

    if (!pm->s3_disabled) {

        pkg = aml_package(4);

        aml_append(pkg, aml_int(1)); /* PM1a_CNT.SLP_TYP */

        aml_append(pkg, aml_int(1)); /* PM1b_CNT.SLP_TYP, FIXME: not impl. */

        aml_append(pkg, aml_int(0)); /* reserved */

        aml_append(pkg, aml_int(0)); /* reserved */

        aml_append(scope, aml_name_decl("_S3", pkg));

    }



    if (!pm->s4_disabled) {

        pkg = aml_package(4);

        aml_append(pkg, aml_int(pm->s4_val)); /* PM1a_CNT.SLP_TYP */

        /* PM1b_CNT.SLP_TYP, FIXME: not impl. */

        aml_append(pkg, aml_int(pm->s4_val));

        aml_append(pkg, aml_int(0)); /* reserved */

        aml_append(pkg, aml_int(0)); /* reserved */

        aml_append(scope, aml_name_decl("_S4", pkg));

    }



    pkg = aml_package(4);

    aml_append(pkg, aml_int(0)); /* PM1a_CNT.SLP_TYP */

    aml_append(pkg, aml_int(0)); /* PM1b_CNT.SLP_TYP not impl. */

    aml_append(pkg, aml_int(0)); /* reserved */

    aml_append(pkg, aml_int(0)); /* reserved */

    aml_append(scope, aml_name_decl("_S5", pkg));

    aml_append(ssdt, scope);



    if (misc->applesmc_io_base) {

        scope = aml_scope("\\_SB.PCI0.ISA");

        dev = aml_device("SMC");



        aml_append(dev, aml_name_decl("_HID", aml_eisaid("APP0001")));

        /* device present, functioning, decoding, not shown in UI */

        aml_append(dev, aml_name_decl("_STA", aml_int(0xB)));



        crs = aml_resource_template();

        aml_append(crs,

            aml_io(AML_DECODE16, misc->applesmc_io_base, misc->applesmc_io_base,

                   0x01, APPLESMC_MAX_DATA_LENGTH)

        );

        aml_append(crs, aml_irq_no_flags(6));

        aml_append(dev, aml_name_decl("_CRS", crs));



        aml_append(scope, dev);

        aml_append(ssdt, scope);

    }



    if (misc->pvpanic_port) {

        scope = aml_scope("\\_SB.PCI0.ISA");



        dev = aml_device("PEVT");

        aml_append(dev, aml_name_decl("_HID", aml_string("QEMU0001")));



        crs = aml_resource_template();

        aml_append(crs,

            aml_io(AML_DECODE16, misc->pvpanic_port, misc->pvpanic_port, 1, 1)

        );

        aml_append(dev, aml_name_decl("_CRS", crs));



        aml_append(dev, aml_operation_region("PEOR", AML_SYSTEM_IO,

                                              misc->pvpanic_port, 1));

        field = aml_field("PEOR", AML_BYTE_ACC, AML_PRESERVE);

        aml_append(field, aml_named_field("PEPT", 8));

        aml_append(dev, field);



        /* device present, functioning, decoding, not shown in UI */

        aml_append(dev, aml_name_decl("_STA", aml_int(0xB)));



        method = aml_method("RDPT", 0);

        aml_append(method, aml_store(aml_name("PEPT"), aml_local(0)));

        aml_append(method, aml_return(aml_local(0)));

        aml_append(dev, method);



        method = aml_method("WRPT", 1);

        aml_append(method, aml_store(aml_arg(0), aml_name("PEPT")));

        aml_append(dev, method);



        aml_append(scope, dev);

        aml_append(ssdt, scope);

    }



    sb_scope = aml_scope("\\_SB");

    {

        /* create PCI0.PRES device and its _CRS to reserve CPU hotplug MMIO */

        dev = aml_device("PCI0." stringify(CPU_HOTPLUG_RESOURCE_DEVICE));

        aml_append(dev, aml_name_decl("_HID", aml_eisaid("PNP0A06")));

        aml_append(dev,

            aml_name_decl("_UID", aml_string("CPU Hotplug resources"))

        );

        /* device present, functioning, decoding, not shown in UI */

        aml_append(dev, aml_name_decl("_STA", aml_int(0xB)));

        crs = aml_resource_template();

        aml_append(crs,

            aml_io(AML_DECODE16, pm->cpu_hp_io_base, pm->cpu_hp_io_base, 1,

                   pm->cpu_hp_io_len)

        );

        aml_append(dev, aml_name_decl("_CRS", crs));

        aml_append(sb_scope, dev);

        /* declare CPU hotplug MMIO region and PRS field to access it */

        aml_append(sb_scope, aml_operation_region(

            "PRST", AML_SYSTEM_IO, pm->cpu_hp_io_base, pm->cpu_hp_io_len));

        field = aml_field("PRST", AML_BYTE_ACC, AML_PRESERVE);

        aml_append(field, aml_named_field("PRS", 256));

        aml_append(sb_scope, field);



        /* build Processor object for each processor */

        for (i = 0; i < acpi_cpus; i++) {

            dev = aml_processor(i, 0, 0, "CP%.02X", i);



            method = aml_method("_MAT", 0);

            aml_append(method, aml_return(aml_call1("CPMA", aml_int(i))));

            aml_append(dev, method);



            method = aml_method("_STA", 0);

            aml_append(method, aml_return(aml_call1("CPST", aml_int(i))));

            aml_append(dev, method);



            method = aml_method("_EJ0", 1);

            aml_append(method,

                aml_return(aml_call2("CPEJ", aml_int(i), aml_arg(0)))

            );

            aml_append(dev, method);



            aml_append(sb_scope, dev);

        }



        /* build this code:

         *   Method(NTFY, 2) {If (LEqual(Arg0, 0x00)) {Notify(CP00, Arg1)} ...}

         */

        /* Arg0 = Processor ID = APIC ID */

        method = aml_method("NTFY", 2);

        for (i = 0; i < acpi_cpus; i++) {

            ifctx = aml_if(aml_equal(aml_arg(0), aml_int(i)));

            aml_append(ifctx,

                aml_notify(aml_name("CP%.02X", i), aml_arg(1))

            );

            aml_append(method, ifctx);

        }

        aml_append(sb_scope, method);



        /* build "Name(CPON, Package() { One, One, ..., Zero, Zero, ... })"

         *

         * Note: The ability to create variable-sized packages was first

         * introduced in ACPI 2.0. ACPI 1.0 only allowed fixed-size packages

         * ith up to 255 elements. Windows guests up to win2k8 fail when

         * VarPackageOp is used.

         */

        pkg = acpi_cpus <= 255 ? aml_package(acpi_cpus) :

                                 aml_varpackage(acpi_cpus);



        for (i = 0; i < acpi_cpus; i++) {

            uint8_t b = test_bit(i, cpu->found_cpus) ? 0x01 : 0x00;

            aml_append(pkg, aml_int(b));

        }

        aml_append(sb_scope, aml_name_decl("CPON", pkg));



        /* build memory devices */

        assert(nr_mem <= ACPI_MAX_RAM_SLOTS);

        scope = aml_scope("\\_SB.PCI0." stringify(MEMORY_HOTPLUG_DEVICE));

        aml_append(scope,

            aml_name_decl(stringify(MEMORY_SLOTS_NUMBER), aml_int(nr_mem))

        );



        crs = aml_resource_template();

        aml_append(crs,

            aml_io(AML_DECODE16, pm->mem_hp_io_base, pm->mem_hp_io_base, 0,

                   pm->mem_hp_io_len)

        );

        aml_append(scope, aml_name_decl("_CRS", crs));



        aml_append(scope, aml_operation_region(

            stringify(MEMORY_HOTPLUG_IO_REGION), AML_SYSTEM_IO,

            pm->mem_hp_io_base, pm->mem_hp_io_len)

        );



        field = aml_field(stringify(MEMORY_HOTPLUG_IO_REGION), AML_DWORD_ACC,

                          AML_PRESERVE);

        aml_append(field, /* read only */

            aml_named_field(stringify(MEMORY_SLOT_ADDR_LOW), 32));

        aml_append(field, /* read only */

            aml_named_field(stringify(MEMORY_SLOT_ADDR_HIGH), 32));

        aml_append(field, /* read only */

            aml_named_field(stringify(MEMORY_SLOT_SIZE_LOW), 32));

        aml_append(field, /* read only */

            aml_named_field(stringify(MEMORY_SLOT_SIZE_HIGH), 32));

        aml_append(field, /* read only */

            aml_named_field(stringify(MEMORY_SLOT_PROXIMITY), 32));

        aml_append(scope, field);



        field = aml_field(stringify(MEMORY_HOTPLUG_IO_REGION), AML_BYTE_ACC,

                          AML_WRITE_AS_ZEROS);

        aml_append(field, aml_reserved_field(160 /* bits, Offset(20) */));

        aml_append(field, /* 1 if enabled, read only */

            aml_named_field(stringify(MEMORY_SLOT_ENABLED), 1));

        aml_append(field,

            /*(read) 1 if has a insert event. (write) 1 to clear event */

            aml_named_field(stringify(MEMORY_SLOT_INSERT_EVENT), 1));

        aml_append(field,

            /* (read) 1 if has a remove event. (write) 1 to clear event */

            aml_named_field(stringify(MEMORY_SLOT_REMOVE_EVENT), 1));

        aml_append(field,

            /* initiates device eject, write only */

            aml_named_field(stringify(MEMORY_SLOT_EJECT), 1));

        aml_append(scope, field);



        field = aml_field(stringify(MEMORY_HOTPLUG_IO_REGION), AML_DWORD_ACC,

                          AML_PRESERVE);

        aml_append(field, /* DIMM selector, write only */

            aml_named_field(stringify(MEMORY_SLOT_SLECTOR), 32));

        aml_append(field, /* _OST event code, write only */

            aml_named_field(stringify(MEMORY_SLOT_OST_EVENT), 32));

        aml_append(field, /* _OST status code, write only */

            aml_named_field(stringify(MEMORY_SLOT_OST_STATUS), 32));

        aml_append(scope, field);



        aml_append(sb_scope, scope);



        for (i = 0; i < nr_mem; i++) {

            #define BASEPATH "\\_SB.PCI0." stringify(MEMORY_HOTPLUG_DEVICE) "."

            const char *s;



            dev = aml_device("MP%02X", i);

            aml_append(dev, aml_name_decl("_UID", aml_string("0x%02X", i)));

            aml_append(dev, aml_name_decl("_HID", aml_eisaid("PNP0C80")));



            method = aml_method("_CRS", 0);

            s = BASEPATH stringify(MEMORY_SLOT_CRS_METHOD);

            aml_append(method, aml_return(aml_call1(s, aml_name("_UID"))));

            aml_append(dev, method);



            method = aml_method("_STA", 0);

            s = BASEPATH stringify(MEMORY_SLOT_STATUS_METHOD);

            aml_append(method, aml_return(aml_call1(s, aml_name("_UID"))));

            aml_append(dev, method);



            method = aml_method("_PXM", 0);

            s = BASEPATH stringify(MEMORY_SLOT_PROXIMITY_METHOD);

            aml_append(method, aml_return(aml_call1(s, aml_name("_UID"))));

            aml_append(dev, method);



            method = aml_method("_OST", 3);

            s = BASEPATH stringify(MEMORY_SLOT_OST_METHOD);

            aml_append(method, aml_return(aml_call4(

                s, aml_name("_UID"), aml_arg(0), aml_arg(1), aml_arg(2)

            )));

            aml_append(dev, method);



            method = aml_method("_EJ0", 1);

            s = BASEPATH stringify(MEMORY_SLOT_EJECT_METHOD);

            aml_append(method, aml_return(aml_call2(

                       s, aml_name("_UID"), aml_arg(0))));

            aml_append(dev, method);



            aml_append(sb_scope, dev);

        }



        /* build Method(MEMORY_SLOT_NOTIFY_METHOD, 2) {

         *     If (LEqual(Arg0, 0x00)) {Notify(MP00, Arg1)} ... }

         */

        method = aml_method(stringify(MEMORY_SLOT_NOTIFY_METHOD), 2);

        for (i = 0; i < nr_mem; i++) {

            ifctx = aml_if(aml_equal(aml_arg(0), aml_int(i)));

            aml_append(ifctx,

                aml_notify(aml_name("MP%.02X", i), aml_arg(1))

            );

            aml_append(method, ifctx);

        }

        aml_append(sb_scope, method);



        {

            Object *pci_host;

            PCIBus *bus = NULL;



            pci_host = acpi_get_i386_pci_host();

            if (pci_host) {

                bus = PCI_HOST_BRIDGE(pci_host)->bus;

            }



            if (bus) {

                Aml *scope = aml_scope("PCI0");

                /* Scan all PCI buses. Generate tables to support hotplug. */

                build_append_pci_bus_devices(scope, bus, pm->pcihp_bridge_en);



                if (misc->tpm_version != TPM_VERSION_UNSPEC) {

                    dev = aml_device("ISA.TPM");

                    aml_append(dev, aml_name_decl("_HID", aml_eisaid("PNP0C31")));

                    aml_append(dev, aml_name_decl("_STA", aml_int(0xF)));

                    crs = aml_resource_template();

                    aml_append(crs, aml_memory32_fixed(TPM_TIS_ADDR_BASE,

                               TPM_TIS_ADDR_SIZE, AML_READ_WRITE));

                    aml_append(crs, aml_irq_no_flags(TPM_TIS_IRQ));

                    aml_append(dev, aml_name_decl("_CRS", crs));

                    aml_append(scope, dev);

                }



                aml_append(sb_scope, scope);

            }

        }

        aml_append(ssdt, sb_scope);

    }



    /* copy AML table into ACPI tables blob and patch header there */

    g_array_append_vals(table_data, ssdt->buf->data, ssdt->buf->len);

    build_header(linker, table_data,

        (void *)(table_data->data + table_data->len - ssdt->buf->len),

        "SSDT", ssdt->buf->len, 1);

    free_aml_allocator();

}
