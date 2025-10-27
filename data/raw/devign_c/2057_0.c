void hmp_info_memory_devices(Monitor *mon, const QDict *qdict)

{

    Error *err = NULL;

    MemoryDeviceInfoList *info_list = qmp_query_memory_devices(&err);

    MemoryDeviceInfoList *info;

    MemoryDeviceInfo *value;

    PCDIMMDeviceInfo *di;



    for (info = info_list; info; info = info->next) {

        value = info->value;



        if (value) {

            switch (value->kind) {

            case MEMORY_DEVICE_INFO_KIND_DIMM:

                di = value->dimm;



                monitor_printf(mon, "Memory device [%s]: \"%s\"\n",

                               MemoryDeviceInfoKind_lookup[value->kind],

                               di->id ? di->id : "");

                monitor_printf(mon, "  addr: 0x%" PRIx64 "\n", di->addr);

                monitor_printf(mon, "  slot: %" PRId64 "\n", di->slot);

                monitor_printf(mon, "  node: %" PRId64 "\n", di->node);

                monitor_printf(mon, "  size: %" PRIu64 "\n", di->size);

                monitor_printf(mon, "  memdev: %s\n", di->memdev);

                monitor_printf(mon, "  hotplugged: %s\n",

                               di->hotplugged ? "true" : "false");

                monitor_printf(mon, "  hotpluggable: %s\n",

                               di->hotpluggable ? "true" : "false");

                break;

            default:

                break;

            }

        }

    }



    qapi_free_MemoryDeviceInfoList(info_list);

}
