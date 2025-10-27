void spapr_drc_detach(sPAPRDRConnector *drc, DeviceState *d, Error **errp)

{

    trace_spapr_drc_detach(spapr_drc_index(drc));



    if (drc->isolation_state != SPAPR_DR_ISOLATION_STATE_ISOLATED) {

        trace_spapr_drc_awaiting_isolated(spapr_drc_index(drc));

        drc->awaiting_release = true;

        return;

    }



    if (spapr_drc_type(drc) != SPAPR_DR_CONNECTOR_TYPE_PCI &&

        drc->allocation_state != SPAPR_DR_ALLOCATION_STATE_UNUSABLE) {

        trace_spapr_drc_awaiting_unusable(spapr_drc_index(drc));

        drc->awaiting_release = true;

        return;

    }



    if (drc->awaiting_allocation) {

        drc->awaiting_release = true;

        trace_spapr_drc_awaiting_allocation(spapr_drc_index(drc));

        return;

    }



    spapr_drc_release(drc);

}
