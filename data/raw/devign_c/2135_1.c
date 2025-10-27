int qdev_build_hotpluggable_device_list(Object *obj, void *opaque)

{

    GSList **list = opaque;

    DeviceState *dev = DEVICE(obj);



    if (dev->realized && object_property_get_bool(obj, "hotpluggable", NULL)) {

        *list = g_slist_append(*list, dev);

    }



    object_child_foreach(obj, qdev_build_hotpluggable_device_list, opaque);

    return 0;

}
