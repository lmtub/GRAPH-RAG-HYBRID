static void register_subpage(AddressSpaceDispatch *d, MemoryRegionSection *section)

{

    subpage_t *subpage;

    hwaddr base = section->offset_within_address_space

        & TARGET_PAGE_MASK;

    MemoryRegionSection *existing = phys_page_find(d->phys_map, base,

                                                   next_map.nodes, next_map.sections);

    MemoryRegionSection subsection = {

        .offset_within_address_space = base,

        .size = int128_make64(TARGET_PAGE_SIZE),

    };

    hwaddr start, end;



    assert(existing->mr->subpage || existing->mr == &io_mem_unassigned);



    if (!(existing->mr->subpage)) {

        subpage = subpage_init(d->as, base);

        subsection.mr = &subpage->iomem;

        phys_page_set(d, base >> TARGET_PAGE_BITS, 1,

                      phys_section_add(&subsection));

    } else {

        subpage = container_of(existing->mr, subpage_t, iomem);

    }

    start = section->offset_within_address_space & ~TARGET_PAGE_MASK;

    end = start + int128_get64(section->size) - 1;

    subpage_register(subpage, start, end, phys_section_add(section));

}
