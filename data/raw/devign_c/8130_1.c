static void device_unparent(Object *obj)

{

    DeviceState *dev = DEVICE(obj);

    BusState *bus;



    if (dev->realized) {

        object_property_set_bool(obj, false, "realized", NULL);

    }

    while (dev->num_child_bus) {

        bus = QLIST_FIRST(&dev->child_bus);

        object_unparent(OBJECT(bus));

    }

    if (dev->parent_bus) {

        bus_remove_child(dev->parent_bus, dev);

        object_unref(OBJECT(dev->parent_bus));

        dev->parent_bus = NULL;

    }



    /* Only send event if the device had been completely realized */

    if (dev->pending_deleted_event) {

        g_assert(dev->canonical_path);



        qapi_event_send_device_deleted(!!dev->id, dev->id, dev->canonical_path,

                                       &error_abort);

        g_free(dev->canonical_path);

        dev->canonical_path = NULL;

    }

}
