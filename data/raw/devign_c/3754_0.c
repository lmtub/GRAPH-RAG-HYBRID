static void numa_stat_memory_devices(uint64_t node_mem[])

{

    MemoryDeviceInfoList *info_list = NULL;

    MemoryDeviceInfoList **prev = &info_list;

    MemoryDeviceInfoList *info;



    qmp_pc_dimm_device_list(qdev_get_machine(), &prev);

    for (info = info_list; info; info = info->next) {

        MemoryDeviceInfo *value = info->value;



        if (value) {

            switch (value->type) {

            case MEMORY_DEVICE_INFO_KIND_DIMM:

                node_mem[value->u.dimm->node] += value->u.dimm->size;

                break;

            default:

                break;

            }

        }

    }

    qapi_free_MemoryDeviceInfoList(info_list);

}
