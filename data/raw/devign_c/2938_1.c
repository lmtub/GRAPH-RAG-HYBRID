void nvdimm_build_acpi(GArray *table_offsets, GArray *table_data,

                       BIOSLinker *linker, GArray *dsm_dma_arrea,

                       uint32_t ram_slots)

{

    GSList *device_list;



    device_list = nvdimm_get_plugged_device_list();



    /* NVDIMM device is plugged. */

    if (device_list) {

        nvdimm_build_nfit(device_list, table_offsets, table_data, linker);

        g_slist_free(device_list);

    }



    /*

     * NVDIMM device is allowed to be plugged only if there is available

     * slot.

     */

    if (ram_slots) {

        nvdimm_build_ssdt(table_offsets, table_data, linker, dsm_dma_arrea,

                          ram_slots);

    }

}
