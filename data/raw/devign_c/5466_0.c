static void nvdimm_build_ssdt(GSList *device_list, GArray *table_offsets,

                              GArray *table_data, BIOSLinker *linker,

                              GArray *dsm_dma_arrea)

{

    Aml *ssdt, *sb_scope, *dev;

    int mem_addr_offset, nvdimm_ssdt;



    acpi_add_table(table_offsets, table_data);



    ssdt = init_aml_allocator();

    acpi_data_push(ssdt->buf, sizeof(AcpiTableHeader));



    sb_scope = aml_scope("\\_SB");



    dev = aml_device("NVDR");



    /*

     * ACPI 6.0: 9.20 NVDIMM Devices:

     *

     * The ACPI Name Space device uses _HID of ACPI0012 to identify the root

     * NVDIMM interface device. Platform firmware is required to contain one

     * such device in _SB scope if NVDIMMs support is exposed by platform to

     * OSPM.

     * For each NVDIMM present or intended to be supported by platform,

     * platform firmware also exposes an ACPI Namespace Device under the

     * root device.

     */

    aml_append(dev, aml_name_decl("_HID", aml_string("ACPI0012")));



    nvdimm_build_common_dsm(dev);



    /* 0 is reserved for root device. */

    nvdimm_build_device_dsm(dev, 0);



    nvdimm_build_nvdimm_devices(device_list, dev);



    aml_append(sb_scope, dev);

    aml_append(ssdt, sb_scope);



    nvdimm_ssdt = table_data->len;



    /* copy AML table into ACPI tables blob and patch header there */

    g_array_append_vals(table_data, ssdt->buf->data, ssdt->buf->len);

    mem_addr_offset = build_append_named_dword(table_data,

                                               NVDIMM_ACPI_MEM_ADDR);



    bios_linker_loader_alloc(linker,

                             NVDIMM_DSM_MEM_FILE, dsm_dma_arrea,

                             sizeof(NvdimmDsmIn), false /* high memory */);

    bios_linker_loader_add_pointer(linker,

        ACPI_BUILD_TABLE_FILE, mem_addr_offset, sizeof(uint32_t),

        NVDIMM_DSM_MEM_FILE, 0);

    build_header(linker, table_data,

        (void *)(table_data->data + nvdimm_ssdt),

        "SSDT", table_data->len - nvdimm_ssdt, 1, NULL, "NVDIMM");

    free_aml_allocator();

}
