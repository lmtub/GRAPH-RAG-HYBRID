static GSList *nvdimm_get_plugged_device_list(void)

{

    GSList *list = NULL;



    object_child_foreach(qdev_get_machine(), nvdimm_plugged_device_list,

                         &list);

    return list;

}
