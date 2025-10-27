static void core_region_del(MemoryListener *listener,

                            MemoryRegionSection *section)

{

    cpu_register_physical_memory_log(section, false);

}
