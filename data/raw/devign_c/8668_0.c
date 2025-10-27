static sPAPRDREntitySense logical_entity_sense(sPAPRDRConnector *drc)

{

    if (drc->dev

        && (drc->allocation_state != SPAPR_DR_ALLOCATION_STATE_UNUSABLE)) {

        return SPAPR_DR_ENTITY_SENSE_PRESENT;

    } else {

        return SPAPR_DR_ENTITY_SENSE_UNUSABLE;

    }

}
