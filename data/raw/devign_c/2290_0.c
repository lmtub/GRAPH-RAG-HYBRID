build_rsdp(GArray *rsdp_table, GArray *linker, unsigned rsdt)

{

    AcpiRsdpDescriptor *rsdp = acpi_data_push(rsdp_table, sizeof *rsdp);



    bios_linker_loader_alloc(linker, ACPI_BUILD_RSDP_FILE, 16,

                             true /* fseg memory */);



    memcpy(&rsdp->signature, "RSD PTR ", sizeof(rsdp->signature));

    memcpy(rsdp->oem_id, ACPI_BUILD_APPNAME6, sizeof(rsdp->oem_id));

    rsdp->length = cpu_to_le32(sizeof(*rsdp));

    rsdp->revision = 0x02;



    /* Point to RSDT */

    rsdp->rsdt_physical_address = cpu_to_le32(rsdt);

    /* Address to be filled by Guest linker */

    bios_linker_loader_add_pointer(linker, ACPI_BUILD_RSDP_FILE,

                                   ACPI_BUILD_TABLE_FILE,

                                   rsdp_table, &rsdp->rsdt_physical_address,

                                   sizeof rsdp->rsdt_physical_address);

    rsdp->checksum = 0;

    /* Checksum to be filled by Guest linker */

    bios_linker_loader_add_checksum(linker, ACPI_BUILD_RSDP_FILE,

                                    rsdp, rsdp, sizeof *rsdp, &rsdp->checksum);



    return rsdp_table;

}
