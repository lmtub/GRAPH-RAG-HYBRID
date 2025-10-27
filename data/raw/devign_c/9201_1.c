static void spapr_memory_pre_plug(HotplugHandler *hotplug_dev, DeviceState *dev,

                                  Error **errp)

{

    PCDIMMDevice *dimm = PC_DIMM(dev);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);

    MemoryRegion *mr = ddc->get_memory_region(dimm);

    uint64_t size = memory_region_size(mr);

    char *mem_dev;



    if (size % SPAPR_MEMORY_BLOCK_SIZE) {

        error_setg(errp, "Hotplugged memory size must be a multiple of "

                      "%lld MB", SPAPR_MEMORY_BLOCK_SIZE / M_BYTE);

        return;

    }



    mem_dev = object_property_get_str(OBJECT(dimm), PC_DIMM_MEMDEV_PROP, NULL);

    if (mem_dev && !kvmppc_is_mem_backend_page_size_ok(mem_dev)) {

        error_setg(errp, "Memory backend has bad page size. "

                   "Use 'memory-backend-file' with correct mem-path.");

        return;

    }

}
