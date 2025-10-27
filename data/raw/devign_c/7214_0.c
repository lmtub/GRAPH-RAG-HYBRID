build_rsdp(GArray *rsdp_table, BIOSLinker *linker, unsigned rsdt_tbl_offset)

{

    AcpiRsdpDescriptor *rsdp = acpi_data_push(rsdp_table, sizeof *rsdp);

    unsigned rsdt_pa_size = sizeof(rsdp->rsdt_physical_address);

    unsigned rsdt_pa_offset =

        (char *)&rsdp->rsdt_physical_address - rsdp_table->data;



    bios_linker_loader_alloc(linker, ACPI_BUILD_RSDP_FILE, rsdp_table, 16,

                             true /* fseg memory */);



    memcpy(&rsdp->signature, "RSD PTR ", sizeof(rsdp->signature));

    memcpy(rsdp->oem_id, ACPI_BUILD_APPNAME6, sizeof(rsdp->oem_id));

    rsdp->length = cpu_to_le32(sizeof(*rsdp));

    rsdp->revision = 0x02;



    /* Address to be filled by Guest linker */

    bios_linker_loader_add_pointer(linker,

        ACPI_BUILD_RSDP_FILE, rsdt_pa_offset, rsdt_pa_size,

        ACPI_BUILD_TABLE_FILE, rsdt_tbl_offset);



    rsdp->checksum = 0;

    /* Checksum to be filled by Guest linker */

    bios_linker_loader_add_checksum(linker, ACPI_BUILD_RSDP_FILE,

                                    rsdp, sizeof *rsdp,

                                    &rsdp->checksum);



    return rsdp_table;

}
