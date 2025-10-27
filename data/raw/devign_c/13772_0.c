sPAPRTCETable *spapr_tce_new_table(DeviceState *owner, uint32_t liobn,

                                   uint64_t bus_offset,

                                   uint32_t page_shift,

                                   uint32_t nb_table,

                                   bool vfio_accel)

{

    sPAPRTCETable *tcet;

    char tmp[64];



    if (spapr_tce_find_by_liobn(liobn)) {

        fprintf(stderr, "Attempted to create TCE table with duplicate"

                " LIOBN 0x%x\n", liobn);

        return NULL;

    }



    if (!nb_table) {

        return NULL;

    }



    tcet = SPAPR_TCE_TABLE(object_new(TYPE_SPAPR_TCE_TABLE));

    tcet->liobn = liobn;

    tcet->bus_offset = bus_offset;

    tcet->page_shift = page_shift;

    tcet->nb_table = nb_table;

    tcet->vfio_accel = vfio_accel;



    snprintf(tmp, sizeof(tmp), "tce-table-%x", liobn);

    object_property_add_child(OBJECT(owner), tmp, OBJECT(tcet), NULL);



    object_property_set_bool(OBJECT(tcet), true, "realized", NULL);



    return tcet;

}
