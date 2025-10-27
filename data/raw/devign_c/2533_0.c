static uint32_t rtas_set_isolation_state(uint32_t idx, uint32_t state)

{

    sPAPRDRConnector *drc = spapr_drc_by_index(idx);

    sPAPRDRConnectorClass *drck;



    if (!drc) {

        return RTAS_OUT_PARAM_ERROR;

    }



    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    return drck->set_isolation_state(drc, state);

}
