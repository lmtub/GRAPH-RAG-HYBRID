BusState *qbus_create(BusType type, size_t size,

                      DeviceState *parent, const char *name)

{

    BusState *bus;



    bus = qemu_mallocz(size);

    bus->type = type;

    bus->parent = parent;

    bus->name = qemu_strdup(name);

    LIST_INIT(&bus->children);

    if (parent) {

        LIST_INSERT_HEAD(&parent->child_bus, bus, sibling);

    }

    return bus;

}
