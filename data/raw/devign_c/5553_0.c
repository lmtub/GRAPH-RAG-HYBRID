sPAPRTCETable *spapr_tce_new_table(DeviceState *owner, uint32_t liobn)

{

    sPAPRTCETable *tcet;

    char tmp[32];



    if (spapr_tce_find_by_liobn(liobn)) {

        error_report("Attempted to create TCE table with duplicate"

                " LIOBN 0x%x", liobn);

        return NULL;

    }



    tcet = SPAPR_TCE_TABLE(object_new(TYPE_SPAPR_TCE_TABLE));

    tcet->liobn = liobn;



    snprintf(tmp, sizeof(tmp), "tce-table-%x", liobn);

    object_property_add_child(OBJECT(owner), tmp, OBJECT(tcet), NULL);



    object_property_set_bool(OBJECT(tcet), true, "realized", NULL);



    return tcet;

}
