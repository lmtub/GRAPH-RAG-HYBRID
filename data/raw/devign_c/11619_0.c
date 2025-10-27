static MemoryRegionSection *address_space_lookup_region(AddressSpaceDispatch *d,

                                                        hwaddr addr,

                                                        bool resolve_subpage)

{

    MemoryRegionSection *section;

    subpage_t *subpage;



    section = phys_page_find(d->phys_map, addr, d->map.nodes, d->map.sections);

    if (resolve_subpage && section->mr->subpage) {

        subpage = container_of(section->mr, subpage_t, iomem);

        section = &d->map.sections[subpage->sub_section[SUBPAGE_IDX(addr)]];

    }

    return section;

}
