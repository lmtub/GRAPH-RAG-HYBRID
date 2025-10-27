void spapr_drc_attach(sPAPRDRConnector *drc, DeviceState *d, void *fdt,

                      int fdt_start_offset, bool coldplug, Error **errp)

{

    trace_spapr_drc_attach(spapr_drc_index(drc));



    if (drc->isolation_state != SPAPR_DR_ISOLATION_STATE_ISOLATED) {

        error_setg(errp, "an attached device is still awaiting release");

        return;

    }

    if (spapr_drc_type(drc) == SPAPR_DR_CONNECTOR_TYPE_PCI) {

        g_assert(drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_USABLE);

    }

    g_assert(fdt || coldplug);



    drc->dr_indicator = SPAPR_DR_INDICATOR_ACTIVE;



    drc->dev = d;

    drc->fdt = fdt;

    drc->fdt_start_offset = fdt_start_offset;

    drc->configured = coldplug;

    /* 'logical' DR resources such as memory/cpus are in some cases treated

     * as a pool of resources from which the guest is free to choose from

     * based on only a count. for resources that can be assigned in this

     * fashion, we must assume the resource is signalled immediately

     * since a single hotplug request might make an arbitrary number of

     * such attached resources available to the guest, as opposed to

     * 'physical' DR resources such as PCI where each device/resource is

     * signalled individually.

     */

    drc->signalled = (spapr_drc_type(drc) != SPAPR_DR_CONNECTOR_TYPE_PCI)

                     ? true : coldplug;



    if (spapr_drc_type(drc) != SPAPR_DR_CONNECTOR_TYPE_PCI) {

        drc->awaiting_allocation = true;

    }



    object_property_add_link(OBJECT(drc), "device",

                             object_get_typename(OBJECT(drc->dev)),

                             (Object **)(&drc->dev),

                             NULL, 0, NULL);

}
