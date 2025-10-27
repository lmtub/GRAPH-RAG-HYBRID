static void nvdimm_build_structure_dcr(GArray *structures, DeviceState *dev)

{

    NvdimmNfitControlRegion *nfit_dcr;

    int slot = object_property_get_int(OBJECT(dev), PC_DIMM_SLOT_PROP,

                                       NULL);

    uint32_t sn = nvdimm_slot_to_sn(slot);



    nfit_dcr = acpi_data_push(structures, sizeof(*nfit_dcr));



    nfit_dcr->type = cpu_to_le16(4 /* NVDIMM Control Region Structure */);

    nfit_dcr->length = cpu_to_le16(sizeof(*nfit_dcr));

    nfit_dcr->dcr_index = cpu_to_le16(nvdimm_slot_to_dcr_index(slot));



    /* vendor: Intel. */

    nfit_dcr->vendor_id = cpu_to_le16(0x8086);

    nfit_dcr->device_id = cpu_to_le16(1);



    /* The _DSM method is following Intel's DSM specification. */

    nfit_dcr->revision_id = cpu_to_le16(1 /* Current Revision supported

                                             in ACPI 6.0 is 1. */);

    nfit_dcr->serial_number = cpu_to_le32(sn);

    nfit_dcr->fic = cpu_to_le16(0x201 /* Format Interface Code. See Chapter

                                         2: NVDIMM Device Specific Method

                                         (DSM) in DSM Spec Rev1.*/);

}
