static MemoryRegion *pc_dimm_get_memory_region(PCDIMMDevice *dimm)

{

    return host_memory_backend_get_memory(dimm->hostmem, &error_abort);

}
