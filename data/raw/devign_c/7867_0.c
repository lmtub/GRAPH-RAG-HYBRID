nvdimm_build_structure_memdev(GArray *structures, DeviceState *dev)

{

    NvdimmNfitMemDev *nfit_memdev;

    uint64_t addr = object_property_get_int(OBJECT(dev), PC_DIMM_ADDR_PROP,

                                            NULL);

    uint64_t size = object_property_get_int(OBJECT(dev), PC_DIMM_SIZE_PROP,

                                            NULL);

    int slot = object_property_get_int(OBJECT(dev), PC_DIMM_SLOT_PROP,

                                            NULL);

    uint32_t handle = nvdimm_slot_to_handle(slot);



    nfit_memdev = acpi_data_push(structures, sizeof(*nfit_memdev));



    nfit_memdev->type = cpu_to_le16(1 /* Memory Device to System Address

                                         Range Map Structure*/);

    nfit_memdev->length = cpu_to_le16(sizeof(*nfit_memdev));

    nfit_memdev->nfit_handle = cpu_to_le32(handle);



    /*

     * associate memory device with System Physical Address Range

     * Structure.

     */

    nfit_memdev->spa_index = cpu_to_le16(nvdimm_slot_to_spa_index(slot));

    /* associate memory device with Control Region Structure. */

    nfit_memdev->dcr_index = cpu_to_le16(nvdimm_slot_to_dcr_index(slot));



    /* The memory region on the device. */

    nfit_memdev->region_len = cpu_to_le64(size);

    nfit_memdev->region_dpa = cpu_to_le64(addr);



    /* Only one interleave for PMEM. */

    nfit_memdev->interleave_ways = cpu_to_le16(1);

}
