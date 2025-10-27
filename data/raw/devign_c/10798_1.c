void acpi_build(AcpiBuildTables *tables, MachineState *machine)

{

    PCMachineState *pcms = PC_MACHINE(machine);

    PCMachineClass *pcmc = PC_MACHINE_GET_CLASS(pcms);

    GArray *table_offsets;

    unsigned facs, dsdt, rsdt, fadt;

    AcpiPmInfo pm;

    AcpiMiscInfo misc;

    AcpiMcfgInfo mcfg;

    Range pci_hole, pci_hole64;

    uint8_t *u;

    size_t aml_len = 0;

    GArray *tables_blob = tables->table_data;

    AcpiSlicOem slic_oem = { .id = NULL, .table_id = NULL };



    acpi_get_pm_info(&pm);

    acpi_get_misc_info(&misc);

    acpi_get_pci_holes(&pci_hole, &pci_hole64);

    acpi_get_slic_oem(&slic_oem);



    table_offsets = g_array_new(false, true /* clear */,

                                        sizeof(uint32_t));

    ACPI_BUILD_DPRINTF("init ACPI tables\n");



    bios_linker_loader_alloc(tables->linker,

                             ACPI_BUILD_TABLE_FILE, tables_blob,

                             64 /* Ensure FACS is aligned */,

                             false /* high memory */);



    /*

     * FACS is pointed to by FADT.

     * We place it first since it's the only table that has alignment

     * requirements.

     */

    facs = tables_blob->len;

    build_facs(tables_blob, tables->linker);



    /* DSDT is pointed to by FADT */

    dsdt = tables_blob->len;

    build_dsdt(tables_blob, tables->linker, &pm, &misc,

               &pci_hole, &pci_hole64, machine);



    /* Count the size of the DSDT and SSDT, we will need it for legacy

     * sizing of ACPI tables.

     */

    aml_len += tables_blob->len - dsdt;



    /* ACPI tables pointed to by RSDT */

    fadt = tables_blob->len;

    acpi_add_table(table_offsets, tables_blob);

    build_fadt(tables_blob, tables->linker, &pm, facs, dsdt,

               slic_oem.id, slic_oem.table_id);

    aml_len += tables_blob->len - fadt;



    acpi_add_table(table_offsets, tables_blob);

    build_madt(tables_blob, tables->linker, pcms);



    if (misc.has_hpet) {

        acpi_add_table(table_offsets, tables_blob);

        build_hpet(tables_blob, tables->linker);

    }

    if (misc.tpm_version != TPM_VERSION_UNSPEC) {

        acpi_add_table(table_offsets, tables_blob);

        build_tpm_tcpa(tables_blob, tables->linker, tables->tcpalog);



        if (misc.tpm_version == TPM_VERSION_2_0) {

            acpi_add_table(table_offsets, tables_blob);

            build_tpm2(tables_blob, tables->linker);

        }

    }

    if (pcms->numa_nodes) {

        acpi_add_table(table_offsets, tables_blob);

        build_srat(tables_blob, tables->linker, machine);

    }

    if (acpi_get_mcfg(&mcfg)) {

        acpi_add_table(table_offsets, tables_blob);

        build_mcfg_q35(tables_blob, tables->linker, &mcfg);

    }

    if (x86_iommu_get_default()) {

        IommuType IOMMUType = x86_iommu_get_type();

        if (IOMMUType == TYPE_AMD) {

            acpi_add_table(table_offsets, tables_blob);

            build_amd_iommu(tables_blob, tables->linker);

        } else if (IOMMUType == TYPE_INTEL) {

            acpi_add_table(table_offsets, tables_blob);

            build_dmar_q35(tables_blob, tables->linker);

        }

    }

    if (pcms->acpi_nvdimm_state.is_enabled) {

        nvdimm_build_acpi(table_offsets, tables_blob, tables->linker,

                          pcms->acpi_nvdimm_state.dsm_mem, machine->ram_slots);

    }



    /* Add tables supplied by user (if any) */

    for (u = acpi_table_first(); u; u = acpi_table_next(u)) {

        unsigned len = acpi_table_len(u);



        acpi_add_table(table_offsets, tables_blob);

        g_array_append_vals(tables_blob, u, len);

    }



    /* RSDT is pointed to by RSDP */

    rsdt = tables_blob->len;

    build_rsdt(tables_blob, tables->linker, table_offsets,

               slic_oem.id, slic_oem.table_id);



    /* RSDP is in FSEG memory, so allocate it separately */

    build_rsdp(tables->rsdp, tables->linker, rsdt);



    /* We'll expose it all to Guest so we want to reduce

     * chance of size changes.

     *

     * We used to align the tables to 4k, but of course this would

     * too simple to be enough.  4k turned out to be too small an

     * alignment very soon, and in fact it is almost impossible to

     * keep the table size stable for all (max_cpus, max_memory_slots)

     * combinations.  So the table size is always 64k for pc-i440fx-2.1

     * and we give an error if the table grows beyond that limit.

     *

     * We still have the problem of migrating from "-M pc-i440fx-2.0".  For

     * that, we exploit the fact that QEMU 2.1 generates _smaller_ tables

     * than 2.0 and we can always pad the smaller tables with zeros.  We can

     * then use the exact size of the 2.0 tables.

     *

     * All this is for PIIX4, since QEMU 2.0 didn't support Q35 migration.

     */

    if (pcmc->legacy_acpi_table_size) {

        /* Subtracting aml_len gives the size of fixed tables.  Then add the

         * size of the PIIX4 DSDT/SSDT in QEMU 2.0.

         */

        int legacy_aml_len =

            pcmc->legacy_acpi_table_size +

            ACPI_BUILD_LEGACY_CPU_AML_SIZE * max_cpus;

        int legacy_table_size =

            ROUND_UP(tables_blob->len - aml_len + legacy_aml_len,

                     ACPI_BUILD_ALIGN_SIZE);

        if (tables_blob->len > legacy_table_size) {

            /* Should happen only with PCI bridges and -M pc-i440fx-2.0.  */

            error_report("Warning: migration may not work.");

        }

        g_array_set_size(tables_blob, legacy_table_size);

    } else {

        /* Make sure we have a buffer in case we need to resize the tables. */

        if (tables_blob->len > ACPI_BUILD_TABLE_SIZE / 2) {

            /* As of QEMU 2.1, this fires with 160 VCPUs and 255 memory slots.  */

            error_report("Warning: ACPI tables are larger than 64k.");

            error_report("Warning: migration may not work.");

            error_report("Warning: please remove CPUs, NUMA nodes, "

                         "memory slots or PCI bridges.");

        }

        acpi_align_size(tables_blob, ACPI_BUILD_TABLE_SIZE);

    }



    acpi_align_size(tables->linker->cmd_blob, ACPI_BUILD_ALIGN_SIZE);



    /* Cleanup memory that's no longer used. */

    g_array_free(table_offsets, true);

}
