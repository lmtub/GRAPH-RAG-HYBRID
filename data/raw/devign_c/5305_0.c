void nvdimm_build_acpi(GArray *table_offsets, GArray *table_data,

                       BIOSLinker *linker, GArray *dsm_dma_arrea)

{

    GSList *device_list;



    /* no NVDIMM device is plugged. */

    device_list = nvdimm_get_plugged_device_list();

    if (!device_list) {

        return;

    }

    nvdimm_build_nfit(device_list, table_offsets, table_data, linker);

    nvdimm_build_ssdt(device_list, table_offsets, table_data, linker,

                      dsm_dma_arrea);

    g_slist_free(device_list);

}
