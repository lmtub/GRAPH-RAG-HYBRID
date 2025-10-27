int qmp_pc_dimm_device_list(Object *obj, void *opaque)

{

    MemoryDeviceInfoList ***prev = opaque;



    if (object_dynamic_cast(obj, TYPE_PC_DIMM)) {

        DeviceState *dev = DEVICE(obj);



        if (dev->realized) {

            MemoryDeviceInfoList *elem = g_new0(MemoryDeviceInfoList, 1);

            MemoryDeviceInfo *info = g_new0(MemoryDeviceInfo, 1);

            PCDIMMDeviceInfo *di = g_new0(PCDIMMDeviceInfo, 1);

            DeviceClass *dc = DEVICE_GET_CLASS(obj);

            PCDIMMDevice *dimm = PC_DIMM(obj);



            if (dev->id) {

                di->has_id = true;

                di->id = g_strdup(dev->id);

            }

            di->hotplugged = dev->hotplugged;

            di->hotpluggable = dc->hotpluggable;

            di->addr = dimm->addr;

            di->slot = dimm->slot;

            di->node = dimm->node;

            di->size = object_property_get_int(OBJECT(dimm), PC_DIMM_SIZE_PROP,

                                               NULL);

            di->memdev = object_get_canonical_path(OBJECT(dimm->hostmem));



            info->dimm = di;

            elem->value = info;

            elem->next = NULL;

            **prev = elem;

            *prev = &elem->next;

        }

    }



    object_child_foreach(obj, qmp_pc_dimm_device_list, opaque);

    return 0;

}
