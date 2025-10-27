build_dmar_q35(GArray *table_data, BIOSLinker *linker)

{

    int dmar_start = table_data->len;



    AcpiTableDmar *dmar;

    AcpiDmarHardwareUnit *drhd;

    AcpiDmarRootPortATS *atsr;

    uint8_t dmar_flags = 0;

    X86IOMMUState *iommu = x86_iommu_get_default();

    AcpiDmarDeviceScope *scope = NULL;

    /* Root complex IOAPIC use one path[0] only */

    size_t ioapic_scope_size = sizeof(*scope) + sizeof(scope->path[0]);



    assert(iommu);

    if (iommu->intr_supported) {

        dmar_flags |= 0x1;      /* Flags: 0x1: INT_REMAP */

    }



    dmar = acpi_data_push(table_data, sizeof(*dmar));

    dmar->host_address_width = VTD_HOST_ADDRESS_WIDTH - 1;

    dmar->flags = dmar_flags;



    /* DMAR Remapping Hardware Unit Definition structure */

    drhd = acpi_data_push(table_data, sizeof(*drhd) + ioapic_scope_size);

    drhd->type = cpu_to_le16(ACPI_DMAR_TYPE_HARDWARE_UNIT);

    drhd->length = cpu_to_le16(sizeof(*drhd) + ioapic_scope_size);

    drhd->flags = ACPI_DMAR_INCLUDE_PCI_ALL;

    drhd->pci_segment = cpu_to_le16(0);

    drhd->address = cpu_to_le64(Q35_HOST_BRIDGE_IOMMU_ADDR);



    /* Scope definition for the root-complex IOAPIC. See VT-d spec

     * 8.3.1 (version Oct. 2014 or later). */

    scope = &drhd->scope[0];

    scope->entry_type = 0x03;   /* Type: 0x03 for IOAPIC */

    scope->length = ioapic_scope_size;

    scope->enumeration_id = ACPI_BUILD_IOAPIC_ID;

    scope->bus = Q35_PSEUDO_BUS_PLATFORM;

    scope->path[0].device = PCI_SLOT(Q35_PSEUDO_DEVFN_IOAPIC);

    scope->path[0].function = PCI_FUNC(Q35_PSEUDO_DEVFN_IOAPIC);



    if (iommu->dt_supported) {

        atsr = acpi_data_push(table_data, sizeof(*atsr));

        atsr->type = cpu_to_le16(ACPI_DMAR_TYPE_ATSR);

        atsr->length = cpu_to_le16(sizeof(*atsr));

        atsr->flags = ACPI_DMAR_ATSR_ALL_PORTS;

        atsr->pci_segment = cpu_to_le16(0);

    }



    build_header(linker, table_data, (void *)(table_data->data + dmar_start),

                 "DMAR", table_data->len - dmar_start, 1, NULL, NULL);

}
