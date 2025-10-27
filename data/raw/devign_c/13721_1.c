void spapr_drc_reset(sPAPRDRConnector *drc)

{

    trace_spapr_drc_reset(spapr_drc_index(drc));



    g_free(drc->ccs);

    drc->ccs = NULL;



    /* immediately upon reset we can safely assume DRCs whose devices

     * are pending removal can be safely removed.

     */

    if (drc->awaiting_release) {

        spapr_drc_release(drc);

    }



    drc->awaiting_allocation = false;



    if (drc->dev) {

        /* A device present at reset is coldplugged */

        drc->isolation_state = SPAPR_DR_ISOLATION_STATE_UNISOLATED;

        if (spapr_drc_type(drc) != SPAPR_DR_CONNECTOR_TYPE_PCI) {

            drc->allocation_state = SPAPR_DR_ALLOCATION_STATE_USABLE;

        }

        drc->dr_indicator = SPAPR_DR_INDICATOR_ACTIVE;

    } else {

        /* Otherwise device is absent, but might be hotplugged */

        drc->isolation_state = SPAPR_DR_ISOLATION_STATE_ISOLATED;

        if (spapr_drc_type(drc) != SPAPR_DR_CONNECTOR_TYPE_PCI) {

            drc->allocation_state = SPAPR_DR_ALLOCATION_STATE_UNUSABLE;

        }

        drc->dr_indicator = SPAPR_DR_INDICATOR_INACTIVE;

    }

}
