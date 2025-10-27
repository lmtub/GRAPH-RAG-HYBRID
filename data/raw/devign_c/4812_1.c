static MemoryRegionSection *address_space_lookup_region(AddressSpaceDispatch *d,

                                                        hwaddr addr,

                                                        bool resolve_subpage)

{

    MemoryRegionSection *section = atomic_read(&d->mru_section);

    subpage_t *subpage;

    bool update;



    if (section && section != &d->map.sections[PHYS_SECTION_UNASSIGNED] &&

        section_covers_addr(section, addr)) {

        update = false;

    } else {

        section = phys_page_find(d, addr);

        update = true;

    }

    if (resolve_subpage && section->mr->subpage) {

        subpage = container_of(section->mr, subpage_t, iomem);

        section = &d->map.sections[subpage->sub_section[SUBPAGE_IDX(addr)]];

    }

    if (update) {

        atomic_set(&d->mru_section, section);

    }

    return section;

}
