static void peripheral_device_del_completion(ReadLineState *rs,

                                             const char *str, size_t len)

{

    Object *peripheral;

    GSList *list = NULL, *item;



    peripheral = object_resolve_path("/machine/peripheral/", NULL);

    if (peripheral == NULL) {

        return;

    }



    object_child_foreach(peripheral, qdev_build_hotpluggable_device_list,

                         &list);



    for (item = list; item; item = g_slist_next(item)) {

        DeviceState *dev = item->data;



        if (dev->id && !strncmp(str, dev->id, len)) {

            readline_add_completion(rs, dev->id);

        }

    }



    g_slist_free(list);

}
