static void destroy_l2_mapping(PhysPageEntry *lp, unsigned level)

{

    unsigned i;

    PhysPageEntry *p;



    if (lp->ptr == PHYS_MAP_NODE_NIL) {

        return;

    }



    p = phys_map_nodes[lp->ptr];

    for (i = 0; i < L2_SIZE; ++i) {

        if (!p[i].is_leaf) {

            destroy_l2_mapping(&p[i], level - 1);

        } else {

            destroy_page_desc(p[i].ptr);

        }

    }

    lp->is_leaf = 0;

    lp->ptr = PHYS_MAP_NODE_NIL;

}
