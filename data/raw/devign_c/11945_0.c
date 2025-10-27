static MemoryRegionSection *phys_page_find(PhysPageEntry lp, hwaddr index,

                                           Node *nodes, MemoryRegionSection *sections)

{

    PhysPageEntry *p;

    int i;



    for (i = P_L2_LEVELS - 1; i >= 0 && !lp.is_leaf; i--) {

        if (lp.ptr == PHYS_MAP_NODE_NIL) {

            return &sections[PHYS_SECTION_UNASSIGNED];

        }

        p = nodes[lp.ptr];

        lp = p[(index >> (i * L2_BITS)) & (L2_SIZE - 1)];

    }

    return &sections[lp.ptr];

}
