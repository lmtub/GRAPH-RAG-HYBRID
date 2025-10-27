void isa_mmio_init(target_phys_addr_t base, target_phys_addr_t size)

{

    MemoryRegion *mr = g_malloc(sizeof(*mr));



    isa_mmio_setup(mr, size);

    memory_region_add_subregion(get_system_memory(), base, mr);

}
