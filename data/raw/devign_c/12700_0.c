static void nvdimm_build_ssdt(GSList *device_list, GArray *table_offsets,

                              GArray *table_data, GArray *linker)

{

    Aml *ssdt, *sb_scope, *dev;



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

    nvdimm_build_device_dsm(dev);



    nvdimm_build_nvdimm_devices(device_list, dev);



    aml_append(sb_scope, dev);



    aml_append(ssdt, sb_scope);

    /* copy AML table into ACPI tables blob and patch header there */

    g_array_append_vals(table_data, ssdt->buf->data, ssdt->buf->len);

    build_header(linker, table_data,

        (void *)(table_data->data + table_data->len - ssdt->buf->len),

        "SSDT", ssdt->buf->len, 1, "NVDIMM");

    free_aml_allocator();

}
