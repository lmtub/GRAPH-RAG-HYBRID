static void spapr_memory_unplug_request(HotplugHandler *hotplug_dev,

                                        DeviceState *dev, Error **errp)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(hotplug_dev);

    Error *local_err = NULL;

    PCDIMMDevice *dimm = PC_DIMM(dev);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);

    MemoryRegion *mr = ddc->get_memory_region(dimm);

    uint64_t size = memory_region_size(mr);

    uint32_t nr_lmbs = size / SPAPR_MEMORY_BLOCK_SIZE;

    uint64_t addr_start, addr;

    int i;

    sPAPRDRConnector *drc;

    sPAPRDRConnectorClass *drck;

    sPAPRDIMMState *ds;



    addr_start = object_property_get_int(OBJECT(dimm), PC_DIMM_ADDR_PROP,

                                         &local_err);

    if (local_err) {

        goto out;

    }



    ds = g_malloc0(sizeof(sPAPRDIMMState));

    ds->nr_lmbs = nr_lmbs;

    ds->dimm = dimm;

    spapr_pending_dimm_unplugs_add(spapr, ds);



    addr = addr_start;

    for (i = 0; i < nr_lmbs; i++) {

        drc = spapr_dr_connector_by_id(SPAPR_DR_CONNECTOR_TYPE_LMB,

                addr / SPAPR_MEMORY_BLOCK_SIZE);

        g_assert(drc);



        drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

        drck->detach(drc, dev, spapr_lmb_release, NULL, errp);

        addr += SPAPR_MEMORY_BLOCK_SIZE;

    }



    drc = spapr_dr_connector_by_id(SPAPR_DR_CONNECTOR_TYPE_LMB,

                                   addr_start / SPAPR_MEMORY_BLOCK_SIZE);

    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    spapr_hotplug_req_remove_by_count_indexed(SPAPR_DR_CONNECTOR_TYPE_LMB,

                                              nr_lmbs,

                                              drck->get_index(drc));

out:

    error_propagate(errp, local_err);

}
