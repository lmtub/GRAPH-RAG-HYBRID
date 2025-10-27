static void bus_add_child(BusState *bus, DeviceState *child)

{

    char name[32];

    BusChild *kid = g_malloc0(sizeof(*kid));



    if (qdev_hotplug) {

        assert(bus->allow_hotplug);

    }



    kid->index = bus->max_index++;

    kid->child = child;

    object_ref(OBJECT(kid->child));



    QTAILQ_INSERT_HEAD(&bus->children, kid, sibling);



    /* This transfers ownership of kid->child to the property.  */

    snprintf(name, sizeof(name), "child[%d]", kid->index);

    object_property_add_link(OBJECT(bus), name,

                             object_get_typename(OBJECT(child)),

                             (Object **)&kid->child,

                             NULL);

}
