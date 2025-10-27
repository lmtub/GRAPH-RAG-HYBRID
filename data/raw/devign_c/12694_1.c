void spapr_drc_attach(sPAPRDRConnector *drc, DeviceState *d, void *fdt,

                      int fdt_start_offset, Error **errp)

{

    trace_spapr_drc_attach(spapr_drc_index(drc));



    if (drc->isolation_state != SPAPR_DR_ISOLATION_STATE_ISOLATED) {

        error_setg(errp, "an attached device is still awaiting release");

        return;

    }

    if (spapr_drc_type(drc) == SPAPR_DR_CONNECTOR_TYPE_PCI) {

        g_assert(drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_USABLE);

    }

    g_assert(fdt);



    drc->dev = d;

    drc->fdt = fdt;

    drc->fdt_start_offset = fdt_start_offset;



    if (spapr_drc_type(drc) != SPAPR_DR_CONNECTOR_TYPE_PCI) {

        drc->awaiting_allocation = true;

    }



    object_property_add_link(OBJECT(drc), "device",

                             object_get_typename(OBJECT(drc->dev)),

                             (Object **)(&drc->dev),

                             NULL, 0, NULL);

}
