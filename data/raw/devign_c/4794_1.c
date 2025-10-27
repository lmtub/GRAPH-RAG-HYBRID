AUXBus *aux_init_bus(DeviceState *parent, const char *name)

{

    AUXBus *bus;



    bus = AUX_BUS(qbus_create(TYPE_AUX_BUS, parent, name));

    bus->bridge = AUXTOI2C(qdev_create(BUS(bus), TYPE_AUXTOI2C));



    /* Memory related. */

    bus->aux_io = g_malloc(sizeof(*bus->aux_io));

    memory_region_init(bus->aux_io, OBJECT(bus), "aux-io", (1 << 20));

    address_space_init(&bus->aux_addr_space, bus->aux_io, "aux-io");

    return bus;

}
