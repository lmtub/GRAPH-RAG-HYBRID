static uint32_t set_allocation_state(sPAPRDRConnector *drc,

                                     sPAPRDRAllocationState state)

{

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    DPRINTFN("drc: %x, set_allocation_state: %x", get_index(drc), state);



    if (state == SPAPR_DR_ALLOCATION_STATE_USABLE) {

        /* if there's no resource/device associated with the DRC, there's

         * no way for us to put it in an allocation state consistent with

         * being 'USABLE'. PAPR 2.7, 13.5.3.4 documents that this should

         * result in an RTAS return code of -3 / "no such indicator"

         */

        if (!drc->dev) {

            return RTAS_OUT_NO_SUCH_INDICATOR;

        }

    }



    if (drc->type != SPAPR_DR_CONNECTOR_TYPE_PCI) {

        drc->allocation_state = state;

        if (drc->awaiting_release &&

            drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_UNUSABLE) {

            DPRINTFN("finalizing device removal");

            drck->detach(drc, DEVICE(drc->dev), drc->detach_cb,

                         drc->detach_cb_opaque, NULL);



        }

    }

    return RTAS_OUT_SUCCESS;

}