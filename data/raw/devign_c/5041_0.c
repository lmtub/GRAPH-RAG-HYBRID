static void core_region_nop(MemoryListener *listener,

                            MemoryRegionSection *section)

{

    cpu_register_physical_memory_log(section, section->readonly);

}
