static NVDIMMDevice *nvdimm_get_device_by_handle(uint32_t handle)

{

    NVDIMMDevice *nvdimm = NULL;

    GSList *list, *device_list = nvdimm_get_plugged_device_list();



    for (list = device_list; list; list = list->next) {

        NVDIMMDevice *nvd = list->data;

        int slot = object_property_get_int(OBJECT(nvd), PC_DIMM_SLOT_PROP,

                                           NULL);



        if (nvdimm_slot_to_handle(slot) == handle) {

            nvdimm = nvd;

            break;

        }

    }



    g_slist_free(device_list);

    return nvdimm;

}
