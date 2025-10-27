static void icp_realize(DeviceState *dev, Error **errp)

{

    ICPState *icp = ICP(dev);

    ICPStateClass *icpc = ICP_GET_CLASS(dev);

    Object *obj;

    Error *err = NULL;



    obj = object_property_get_link(OBJECT(dev), ICP_PROP_XICS, &err);

    if (!obj) {

        error_setg(errp, "%s: required link '" ICP_PROP_XICS "' not found: %s",

                   __func__, error_get_pretty(err));

        return;

    }



    icp->xics = XICS_FABRIC(obj);



    if (icpc->realize) {

        icpc->realize(dev, errp);

    }



    qemu_register_reset(icp_reset, dev);

}
