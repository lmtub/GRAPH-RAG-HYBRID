static int nvdimm_plugged_device_list(Object *obj, void *opaque)

{

    GSList **list = opaque;



    if (object_dynamic_cast(obj, TYPE_NVDIMM)) {

        *list = g_slist_append(*list, DEVICE(obj));

    }



    object_child_foreach(obj, nvdimm_plugged_device_list, opaque);

    return 0;

}
