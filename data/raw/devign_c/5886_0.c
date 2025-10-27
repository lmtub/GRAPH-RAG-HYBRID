static uint32_t rtas_set_dr_indicator(uint32_t idx, uint32_t state)

{

    sPAPRDRConnector *drc = spapr_drc_by_index(idx);



    if (!drc) {

        return RTAS_OUT_PARAM_ERROR;

    }



    trace_spapr_drc_set_dr_indicator(idx, state);

    drc->dr_indicator = state;

    return RTAS_OUT_SUCCESS;

}
