static bool vfio_prereg_listener_skipped_section(MemoryRegionSection *section)

{

    if (memory_region_is_iommu(section->mr)) {

        hw_error("Cannot possibly preregister IOMMU memory");

    }



    return !memory_region_is_ram(section->mr) ||

            memory_region_is_skip_dump(section->mr);

}
