sPAPRDRConnector *spapr_dr_connector_new(Object *owner, const char *type,

                                         uint32_t id)

{

    sPAPRDRConnector *drc = SPAPR_DR_CONNECTOR(object_new(type));

    char *prop_name;



    drc->id = id;

    drc->owner = owner;

    prop_name = g_strdup_printf("dr-connector[%"PRIu32"]",

                                spapr_drc_index(drc));

    object_property_add_child(owner, prop_name, OBJECT(drc), NULL);

    object_property_set_bool(OBJECT(drc), true, "realized", NULL);

    g_free(prop_name);



    /* PCI slot always start in a USABLE state, and stay there */

    if (spapr_drc_type(drc) == SPAPR_DR_CONNECTOR_TYPE_PCI) {

        drc->allocation_state = SPAPR_DR_ALLOCATION_STATE_USABLE;

    }



    return drc;

}
