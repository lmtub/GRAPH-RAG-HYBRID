build_dsdt(GArray *table_data, BIOSLinker *linker,

           AcpiPmInfo *pm, AcpiMiscInfo *misc,

           Range *pci_hole, Range *pci_hole64, MachineState *machine)

{

    CrsRangeEntry *entry;

    Aml *dsdt, *sb_scope, *scope, *dev, *method, *field, *pkg, *crs;

    CrsRangeSet crs_range_set;

    PCMachineState *pcms = PC_MACHINE(machine);

    PCMachineClass *pcmc = PC_MACHINE_GET_CLASS(machine);

    uint32_t nr_mem = machine->ram_slots;

    int root_bus_limit = 0xFF;

    PCIBus *bus = NULL;

    int i;



    dsdt = init_aml_allocator();



    /* Reserve space for header */

    acpi_data_push(dsdt->buf, sizeof(AcpiTableHeader));



    build_dbg_aml(dsdt);

    if (misc->is_piix4) {

        sb_scope = aml_scope("_SB");

        dev = aml_device("PCI0");

        aml_append(dev, aml_name_decl("_HID", aml_eisaid("PNP0A03")));

        aml_append(dev, aml_name_decl("_ADR", aml_int(0)));

        aml_append(dev, aml_name_decl("_UID", aml_int(1)));

        aml_append(sb_scope, dev);

        aml_append(dsdt, sb_scope);



        build_hpet_aml(dsdt);

        build_piix4_pm(dsdt);

        build_piix4_isa_bridge(dsdt);

        build_isa_devices_aml(dsdt);

        build_piix4_pci_hotplug(dsdt);

        build_piix4_pci0_int(dsdt);

    } else {

        sb_scope = aml_scope("_SB");

        dev = aml_device("PCI0");

        aml_append(dev, aml_name_decl("_HID", aml_eisaid("PNP0A08")));

        aml_append(dev, aml_name_decl("_CID", aml_eisaid("PNP0A03")));

        aml_append(dev, aml_name_decl("_ADR", aml_int(0)));

        aml_append(dev, aml_name_decl("_UID", aml_int(1)));

        aml_append(dev, build_q35_osc_method());

        aml_append(sb_scope, dev);

        aml_append(dsdt, sb_scope);



        build_hpet_aml(dsdt);

        build_q35_isa_bridge(dsdt);

        build_isa_devices_aml(dsdt);

        build_q35_pci0_int(dsdt);

    }



    if (pcmc->legacy_cpu_hotplug) {

        build_legacy_cpu_hotplug_aml(dsdt, machine, pm->cpu_hp_io_base);

    } else {

        CPUHotplugFeatures opts = {

            .apci_1_compatible = true, .has_legacy_cphp = true

        };

        build_cpus_aml(dsdt, machine, opts, pm->cpu_hp_io_base,

                       "\\_SB.PCI0", "\\_GPE._E02");

    }

    build_memory_hotplug_aml(dsdt, nr_mem, "\\_SB.PCI0", "\\_GPE._E03");



    scope =  aml_scope("_GPE");

    {

        aml_append(scope, aml_name_decl("_HID", aml_string("ACPI0006")));



        if (misc->is_piix4) {

            method = aml_method("_E01", 0, AML_NOTSERIALIZED);

            aml_append(method,

                aml_acquire(aml_name("\\_SB.PCI0.BLCK"), 0xFFFF));

            aml_append(method, aml_call0("\\_SB.PCI0.PCNT"));

            aml_append(method, aml_release(aml_name("\\_SB.PCI0.BLCK")));

            aml_append(scope, method);

        }



        if (pcms->acpi_nvdimm_state.is_enabled) {

            method = aml_method("_E04", 0, AML_NOTSERIALIZED);

            aml_append(method, aml_notify(aml_name("\\_SB.NVDR"),

                                          aml_int(0x80)));

            aml_append(scope, method);

        }

    }

    aml_append(dsdt, scope);



    crs_range_set_init(&crs_range_set);

    bus = PC_MACHINE(machine)->bus;

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

            if (pci_bus_is_express(bus)) {

                aml_append(dev, build_q35_osc_method());

            }



            if (numa_node != NUMA_NODE_UNASSIGNED) {

                aml_append(dev, aml_name_decl("_PXM", aml_int(numa_node)));

            }



            aml_append(dev, build_prt(false));

            crs = build_crs(PCI_HOST_BRIDGE(BUS(bus)->parent), &crs_range_set);

            aml_append(dev, aml_name_decl("_CRS", crs));

            aml_append(scope, dev);

            aml_append(dsdt, scope);

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



    crs_replace_with_free_ranges(crs_range_set.io_ranges, 0x0D00, 0xFFFF);

    for (i = 0; i < crs_range_set.io_ranges->len; i++) {

        entry = g_ptr_array_index(crs_range_set.io_ranges, i);

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



    crs_replace_with_free_ranges(crs_range_set.mem_ranges,

                                 range_lob(pci_hole),

                                 range_upb(pci_hole));

    for (i = 0; i < crs_range_set.mem_ranges->len; i++) {

        entry = g_ptr_array_index(crs_range_set.mem_ranges, i);

        aml_append(crs,

            aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED, AML_MAX_FIXED,

                             AML_NON_CACHEABLE, AML_READ_WRITE,

                             0, entry->base, entry->limit,

                             0, entry->limit - entry->base + 1));

    }



    if (!range_is_empty(pci_hole64)) {

        crs_replace_with_free_ranges(crs_range_set.mem_64bit_ranges,

                                     range_lob(pci_hole64),

                                     range_upb(pci_hole64));

        for (i = 0; i < crs_range_set.mem_64bit_ranges->len; i++) {

            entry = g_ptr_array_index(crs_range_set.mem_64bit_ranges, i);

            aml_append(crs,

                       aml_qword_memory(AML_POS_DECODE, AML_MIN_FIXED,

                                        AML_MAX_FIXED,

                                        AML_CACHEABLE, AML_READ_WRITE,

                                        0, entry->base, entry->limit,

                                        0, entry->limit - entry->base + 1));

        }

    }



    if (misc->tpm_version != TPM_VERSION_UNSPEC) {

        aml_append(crs, aml_memory32_fixed(TPM_TIS_ADDR_BASE,

                   TPM_TIS_ADDR_SIZE, AML_READ_WRITE));

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



    crs_range_set_free(&crs_range_set);



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

    aml_append(dsdt, scope);



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

    aml_append(dsdt, scope);



    /* create fw_cfg node, unconditionally */

    {

        /* when using port i/o, the 8-bit data register *always* overlaps

         * with half of the 16-bit control register. Hence, the total size

         * of the i/o region used is FW_CFG_CTL_SIZE; when using DMA, the

         * DMA control register is located at FW_CFG_DMA_IO_BASE + 4 */

        uint8_t io_size = object_property_get_bool(OBJECT(pcms->fw_cfg),

                                                   "dma_enabled", NULL) ?

                          ROUND_UP(FW_CFG_CTL_SIZE, 4) + sizeof(dma_addr_t) :

                          FW_CFG_CTL_SIZE;



        scope = aml_scope("\\_SB.PCI0");

        dev = aml_device("FWCF");



        aml_append(dev, aml_name_decl("_HID", aml_string("QEMU0002")));



        /* device present, functioning, decoding, not shown in UI */

        aml_append(dev, aml_name_decl("_STA", aml_int(0xB)));



        crs = aml_resource_template();

        aml_append(crs,

            aml_io(AML_DECODE16, FW_CFG_IO_BASE, FW_CFG_IO_BASE, 0x01, io_size)

        );

        aml_append(dev, aml_name_decl("_CRS", crs));



        aml_append(scope, dev);

        aml_append(dsdt, scope);

    }



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

        aml_append(dsdt, scope);

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

                                              aml_int(misc->pvpanic_port), 1));

        field = aml_field("PEOR", AML_BYTE_ACC, AML_NOLOCK, AML_PRESERVE);

        aml_append(field, aml_named_field("PEPT", 8));

        aml_append(dev, field);



        /* device present, functioning, decoding, shown in UI */

        aml_append(dev, aml_name_decl("_STA", aml_int(0xF)));



        method = aml_method("RDPT", 0, AML_NOTSERIALIZED);

        aml_append(method, aml_store(aml_name("PEPT"), aml_local(0)));

        aml_append(method, aml_return(aml_local(0)));

        aml_append(dev, method);



        method = aml_method("WRPT", 1, AML_NOTSERIALIZED);

        aml_append(method, aml_store(aml_arg(0), aml_name("PEPT")));

        aml_append(dev, method);



        aml_append(scope, dev);

        aml_append(dsdt, scope);

    }



    sb_scope = aml_scope("\\_SB");

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

                /*

                    FIXME: TPM_TIS_IRQ=5 conflicts with PNP0C0F irqs,

                    Rewrite to take IRQ from TPM device model and

                    fix default IRQ value there to use some unused IRQ

                 */

                /* aml_append(crs, aml_irq_no_flags(TPM_TIS_IRQ)); */

                aml_append(dev, aml_name_decl("_CRS", crs));

                aml_append(scope, dev);

            }



            aml_append(sb_scope, scope);

        }

    }

    aml_append(dsdt, sb_scope);



    /* copy AML table into ACPI tables blob and patch header there */

    g_array_append_vals(table_data, dsdt->buf->data, dsdt->buf->len);

    build_header(linker, table_data,

        (void *)(table_data->data + table_data->len - dsdt->buf->len),

        "DSDT", dsdt->buf->len, 1, NULL, NULL);

    free_aml_allocator();

}
