static void destroy_page_desc(uint16_t section_index)

{

    MemoryRegionSection *section = &phys_sections[section_index];

    MemoryRegion *mr = section->mr;



    if (mr->subpage) {

        subpage_t *subpage = container_of(mr, subpage_t, iomem);

        memory_region_destroy(&subpage->iomem);

        g_free(subpage);

    }

}
