void qbus_create_inplace(BusState *bus, const char *typename,

                         DeviceState *parent, const char *name)

{

    object_initialize(bus, typename);

    qbus_realize(bus, parent, name);

}
