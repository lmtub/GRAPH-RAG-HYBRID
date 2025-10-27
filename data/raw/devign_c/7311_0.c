void macio_nvram_setup_bar(MacIONVRAMState *s, MemoryRegion *bar,

                           target_phys_addr_t mem_base)

{

    memory_region_add_subregion(bar, mem_base, &s->mem);

}
