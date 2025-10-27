static void phys_page_set_level(PhysPageEntry *lp, hwaddr *index,

                                hwaddr *nb, uint16_t leaf,

                                int level)

{

    PhysPageEntry *p;

    int i;

    hwaddr step = (hwaddr)1 << (level * P_L2_BITS);



    if (lp->skip && lp->ptr == PHYS_MAP_NODE_NIL) {

        lp->ptr = phys_map_node_alloc();

        p = next_map.nodes[lp->ptr];

        if (level == 0) {

            for (i = 0; i < P_L2_SIZE; i++) {

                p[i].skip = 0;

                p[i].ptr = PHYS_SECTION_UNASSIGNED;

            }

        }

    } else {

        p = next_map.nodes[lp->ptr];

    }

    lp = &p[(*index >> (level * P_L2_BITS)) & (P_L2_SIZE - 1)];



    while (*nb && lp < &p[P_L2_SIZE]) {

        if ((*index & (step - 1)) == 0 && *nb >= step) {

            lp->skip = 0;

            lp->ptr = leaf;

            *index += step;

            *nb -= step;

        } else {

            phys_page_set_level(lp, index, nb, leaf, level - 1);

        }

        ++lp;

    }

}
