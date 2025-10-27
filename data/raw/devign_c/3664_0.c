static void ics_base_realize(DeviceState *dev, Error **errp)

{

    ICSStateClass *icsc = ICS_BASE_GET_CLASS(dev);

    ICSState *ics = ICS_BASE(dev);

    Object *obj;

    Error *err = NULL;



    obj = object_property_get_link(OBJECT(dev), ICS_PROP_XICS, &err);

    if (!obj) {

        error_setg(errp, "%s: required link '" ICS_PROP_XICS "' not found: %s",

                   __func__, error_get_pretty(err));

        return;

    }

    ics->xics = XICS_FABRIC(obj);





    if (icsc->realize) {

        icsc->realize(dev, errp);

    }

}
