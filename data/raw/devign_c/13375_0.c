static void unrealize(DeviceState *d, Error **errp)

{

    sPAPRDRConnector *drc = SPAPR_DR_CONNECTOR(d);

    Object *root_container;

    char name[256];

    Error *err = NULL;



    trace_spapr_drc_unrealize(spapr_drc_index(drc));

    root_container = container_get(object_get_root(), DRC_CONTAINER_PATH);

    snprintf(name, sizeof(name), "%x", spapr_drc_index(drc));

    object_property_del(root_container, name, &err);

    if (err) {

        error_report_err(err);

        object_unref(OBJECT(drc));

    }

}
