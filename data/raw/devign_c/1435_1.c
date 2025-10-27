static void set_vlan(Object *obj, Visitor *v, void *opaque,

                     const char *name, Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    Property *prop = opaque;

    NICPeers *peers_ptr = qdev_get_prop_ptr(dev, prop);

    NetClientState **ptr = &peers_ptr->ncs[0];

    Error *local_err = NULL;

    int32_t id;

    NetClientState *hubport;



    if (dev->realized) {

        qdev_prop_set_after_realize(dev, name, errp);





    visit_type_int32(v, &id, name, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);



    if (id == -1) {

        *ptr = NULL;









    hubport = net_hub_port_find(id);

    if (!hubport) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE,

                  name, prop->info->name);



    *ptr = hubport;
