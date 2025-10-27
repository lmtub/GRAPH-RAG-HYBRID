static bool vhost_section(MemoryRegionSection *section)

{

    return memory_region_is_ram(section->mr);

}
