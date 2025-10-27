build_dmar_q35(GArray *table_data, GArray *linker)

{

    int dmar_start = table_data->len;



    AcpiTableDmar *dmar;

    AcpiDmarHardwareUnit *drhd;



    dmar = acpi_data_push(table_data, sizeof(*dmar));

    dmar->host_address_width = VTD_HOST_ADDRESS_WIDTH - 1;

    dmar->flags = 0;    /* No intr_remap for now */



    /* DMAR Remapping Hardware Unit Definition structure */

    drhd = acpi_data_push(table_data, sizeof(*drhd));

    drhd->type = cpu_to_le16(ACPI_DMAR_TYPE_HARDWARE_UNIT);

    drhd->length = cpu_to_le16(sizeof(*drhd));   /* No device scope now */

    drhd->flags = ACPI_DMAR_INCLUDE_PCI_ALL;

    drhd->pci_segment = cpu_to_le16(0);

    drhd->address = cpu_to_le64(Q35_HOST_BRIDGE_IOMMU_ADDR);



    build_header(linker, table_data, (void *)(table_data->data + dmar_start),

                 "DMAR", table_data->len - dmar_start, 1, NULL);

}
