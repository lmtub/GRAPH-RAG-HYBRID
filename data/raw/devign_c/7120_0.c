static void memory_region_prepare_ram_addr(MemoryRegion *mr)

{

    if (mr->backend_registered) {

        return;

    }



    mr->destructor = memory_region_destructor_iomem;

    mr->ram_addr = cpu_register_io_memory(memory_region_read_thunk,

                                          memory_region_write_thunk,

                                          mr);

    mr->backend_registered = true;

}
