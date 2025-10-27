build_ssdt(GArray *table_data, GArray *linker,

           AcpiCpuInfo *cpu, AcpiPmInfo *pm, AcpiMiscInfo *misc,

           PcPciInfo *pci, PcGuestInfo *guest_info)

{

    int acpi_cpus = MIN(0xff, guest_info->apic_id_limit);

    int ssdt_start = table_data->len;

    uint8_t *ssdt_ptr;

    int i;



    /* Copy header and patch values in the S3_ / S4_ / S5_ packages */

    ssdt_ptr = acpi_data_push(table_data, sizeof(ssdp_misc_aml));

    memcpy(ssdt_ptr, ssdp_misc_aml, sizeof(ssdp_misc_aml));

    if (pm->s3_disabled) {

        ssdt_ptr[acpi_s3_name[0]] = 'X';

    }

    if (pm->s4_disabled) {

        ssdt_ptr[acpi_s4_name[0]] = 'X';

    } else {

        ssdt_ptr[acpi_s4_pkg[0] + 1] = ssdt_ptr[acpi_s4_pkg[0] + 3] =

            pm->s4_val;

    }



    patch_pci_windows(pci, ssdt_ptr, sizeof(ssdp_misc_aml));



    ACPI_BUILD_SET_LE(ssdt_ptr, sizeof(ssdp_misc_aml),

                      ssdt_isa_pest[0], 16, misc->pvpanic_port);



    {

        GArray *sb_scope = build_alloc_array();

        uint8_t op = 0x10; /* ScopeOp */



        build_append_nameseg(sb_scope, "_SB_");



        /* build Processor object for each processor */

        for (i = 0; i < acpi_cpus; i++) {

            uint8_t *proc = acpi_data_push(sb_scope, ACPI_PROC_SIZEOF);

            memcpy(proc, ACPI_PROC_AML, ACPI_PROC_SIZEOF);

            proc[ACPI_PROC_OFFSET_CPUHEX] = acpi_get_hex(i >> 4);

            proc[ACPI_PROC_OFFSET_CPUHEX+1] = acpi_get_hex(i);

            proc[ACPI_PROC_OFFSET_CPUID1] = i;

            proc[ACPI_PROC_OFFSET_CPUID2] = i;

        }



        /* build this code:

         *   Method(NTFY, 2) {If (LEqual(Arg0, 0x00)) {Notify(CP00, Arg1)} ...}

         */

        /* Arg0 = Processor ID = APIC ID */

        build_append_notify_method(sb_scope, "NTFY", "CP%0.02X", acpi_cpus);



        /* build "Name(CPON, Package() { One, One, ..., Zero, Zero, ... })" */

        build_append_byte(sb_scope, 0x08); /* NameOp */

        build_append_nameseg(sb_scope, "CPON");



        {

            GArray *package = build_alloc_array();

            uint8_t op = 0x12; /* PackageOp */



            build_append_byte(package, acpi_cpus); /* NumElements */

            for (i = 0; i < acpi_cpus; i++) {

                uint8_t b = test_bit(i, cpu->found_cpus) ? 0x01 : 0x00;

                build_append_byte(package, b);

            }



            build_package(package, op, 2);

            build_append_array(sb_scope, package);

            build_free_array(package);

        }



        {

            AcpiBuildPciBusHotplugState hotplug_state;

            Object *pci_host;

            PCIBus *bus = NULL;

            bool ambiguous;



            pci_host = object_resolve_path_type("", TYPE_PCI_HOST_BRIDGE, &ambiguous);

            if (!ambiguous && pci_host) {

                bus = PCI_HOST_BRIDGE(pci_host)->bus;

            }



            build_pci_bus_state_init(&hotplug_state, NULL);



            if (bus) {

                /* Scan all PCI buses. Generate tables to support hotplug. */

                pci_for_each_bus_depth_first(bus, build_pci_bus_begin,

                                             build_pci_bus_end, &hotplug_state);

            }



            build_append_array(sb_scope, hotplug_state.device_table);

            build_pci_bus_state_cleanup(&hotplug_state);

        }



        build_package(sb_scope, op, 3);

        build_append_array(table_data, sb_scope);

        build_free_array(sb_scope);

    }



    build_header(linker, table_data,

                 (void *)(table_data->data + ssdt_start),

                 ACPI_SSDT_SIGNATURE, table_data->len - ssdt_start, 1);

}
