BusState *qbus_create(BusInfo *info, DeviceState *parent, const char *name)

{

    BusState *bus;

    char *buf;

    int i,len;



    bus = qemu_mallocz(info->size);

    bus->info = info;

    bus->parent = parent;



    if (name) {

        /* use supplied name */

        bus->name = qemu_strdup(name);

    } else if (parent && parent->id) {

        /* parent device has id -> use it for bus name */

        len = strlen(parent->id) + 16;

        buf = qemu_malloc(len);

        snprintf(buf, len, "%s.%d", parent->id, parent->num_child_bus);

        bus->name = buf;

    } else {

        /* no id -> use lowercase bus type for bus name */

        len = strlen(info->name) + 16;

        buf = qemu_malloc(len);

        len = snprintf(buf, len, "%s.%d", info->name,

                       parent ? parent->num_child_bus : 0);

        for (i = 0; i < len; i++)

            buf[i] = qemu_tolower(buf[i]);

        bus->name = buf;

    }



    LIST_INIT(&bus->children);

    if (parent) {

        LIST_INSERT_HEAD(&parent->child_bus, bus, sibling);

        parent->num_child_bus++;

    }

    return bus;

}
