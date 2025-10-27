static uint32_t set_allocation_state(sPAPRDRConnector *drc,

                                     sPAPRDRAllocationState state)

{

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    trace_spapr_drc_set_allocation_state(get_index(drc), state);



    if (state == SPAPR_DR_ALLOCATION_STATE_USABLE) {

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

            drc->awaiting_allocation_skippable = true;

            return RTAS_OUT_NO_SUCH_INDICATOR;

        }

    }



    if (drc->type != SPAPR_DR_CONNECTOR_TYPE_PCI) {

        drc->allocation_state = state;

        if (drc->awaiting_release &&

            drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_UNUSABLE) {

            trace_spapr_drc_set_allocation_state_finalizing(get_index(drc));

            drck->detach(drc, DEVICE(drc->dev), drc->detach_cb,

                         drc->detach_cb_opaque, NULL);

        } else if (drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_USABLE) {

            drc->awaiting_allocation = false;

        }

    }

    return RTAS_OUT_SUCCESS;

}
