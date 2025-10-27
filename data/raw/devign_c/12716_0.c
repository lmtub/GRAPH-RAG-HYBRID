void memory_region_init_ram_ptr(MemoryRegion *mr,

                                const char *name,

                                uint64_t size,

                                void *ptr)

{

    memory_region_init(mr, name, size);

    mr->ram = true;

    mr->terminates = true;

    mr->destructor = memory_region_destructor_ram_from_ptr;

    mr->ram_addr = qemu_ram_alloc_from_ptr(size, ptr, mr);

    mr->backend_registered = true;

}
