void spapr_lmb_release(DeviceState *dev)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_hotplug_handler(dev));

    PCDIMMDevice *dimm = PC_DIMM(dev);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);

    MemoryRegion *mr = ddc->get_memory_region(dimm);

    sPAPRDIMMState *ds = spapr_pending_dimm_unplugs_find(spapr, PC_DIMM(dev));



    /* This information will get lost if a migration occurs

     * during the unplug process. In this case recover it. */

    if (ds == NULL) {

        ds = spapr_recover_pending_dimm_state(spapr, PC_DIMM(dev));

        g_assert(ds);

        /* The DRC being examined by the caller at least must be counted */

        g_assert(ds->nr_lmbs);

    }



    if (--ds->nr_lmbs) {

        return;

    }



    spapr_pending_dimm_unplugs_remove(spapr, ds);



    /*

     * Now that all the LMBs have been removed by the guest, call the

     * pc-dimm unplug handler to cleanup up the pc-dimm device.

     */

    pc_dimm_memory_unplug(dev, &spapr->hotplug_memory, mr);

    object_unparent(OBJECT(dev));

}
