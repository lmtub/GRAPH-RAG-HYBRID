static uint32_t drc_set_usable(sPAPRDRConnector *drc)

{

    /* if there's no resource/device associated with the DRC, there's

     * no way for us to put it in an allocation state consistent with

     * being 'USABLE'. PAPR 2.7, 13.5.3.4 documents that this should

     * result in an RTAS return code of -3 / "no such indicator"

     */

    if (!drc->dev) {

        return RTAS_OUT_NO_SUCH_INDICATOR;

    }

    if (drc->awaiting_release && drc->awaiting_allocation) {

        /* kernel is acknowledging a previous hotplug event

         * while we are already removing it.

         * it's safe to ignore awaiting_allocation here since we know the

         * situation is predicated on the guest either already having done

         * so (boot-time hotplug), or never being able to acquire in the

         * first place (hotplug followed by immediate unplug).

         */

        return RTAS_OUT_NO_SUCH_INDICATOR;

    }



    drc->allocation_state = SPAPR_DR_ALLOCATION_STATE_USABLE;

    drc->awaiting_allocation = false;



    return RTAS_OUT_SUCCESS;

}
