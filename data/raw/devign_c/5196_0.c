void spapr_drc_reset(sPAPRDRConnector *drc)

{

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    trace_spapr_drc_reset(spapr_drc_index(drc));



    g_free(drc->ccs);

    drc->ccs = NULL;



    /* immediately upon reset we can safely assume DRCs whose devices

     * are pending removal can be safely removed.

     */

    if (drc->unplug_requested) {

        spapr_drc_release(drc);

    }



    if (drc->dev) {

        /* A device present at reset is ready to go, same as coldplugged */

        drc->state = drck->ready_state;

    } else {

        drc->state = drck->empty_state;

    }

}
