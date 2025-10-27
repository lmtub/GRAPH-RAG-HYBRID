static uint32_t drc_set_unusable(sPAPRDRConnector *drc)

{

    drc->allocation_state = SPAPR_DR_ALLOCATION_STATE_UNUSABLE;

    if (drc->awaiting_release) {

        uint32_t drc_index = spapr_drc_index(drc);

        trace_spapr_drc_set_allocation_state_finalizing(drc_index);

        spapr_drc_detach(drc);

    }



    return RTAS_OUT_SUCCESS;

}
