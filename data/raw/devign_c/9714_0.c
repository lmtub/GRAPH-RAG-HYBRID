static bool vhost_section(MemoryRegionSection *section)

{

    return section->address_space == get_system_memory()

        && memory_region_is_ram(section->mr);

}
