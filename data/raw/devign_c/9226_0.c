static void spapr_drc_release(sPAPRDRConnector *drc)

{

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    drck->release(drc->dev);



    drc->awaiting_release = false;

    g_free(drc->fdt);

    drc->fdt = NULL;

    drc->fdt_start_offset = 0;

    object_property_del(OBJECT(drc), "device", &error_abort);

    drc->dev = NULL;

}
