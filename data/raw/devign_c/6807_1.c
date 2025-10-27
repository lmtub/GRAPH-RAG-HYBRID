static uint16_t phys_section_add(MemoryRegionSection *section)
{
    if (phys_sections_nb == phys_sections_nb_alloc) {
        phys_sections_nb_alloc = MAX(phys_sections_nb_alloc * 2, 16);
        phys_sections = g_renew(MemoryRegionSection, phys_sections,
                                phys_sections_nb_alloc);
    }
    phys_sections[phys_sections_nb] = *section;
    return phys_sections_nb++;
}