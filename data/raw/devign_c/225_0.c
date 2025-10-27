static uint16_t *phys_page_find_alloc(target_phys_addr_t index, int alloc)

{

    PhysPageEntry *lp, *p;

    int i, j;



    lp = &phys_map;



    /* Level 1..N.  */

    for (i = P_L2_LEVELS - 1; i >= 0; i--) {

        if (lp->u.node == NULL) {

            if (!alloc) {

                return NULL;

            }

            lp->u.node = p = g_malloc0(sizeof(PhysPageEntry) * L2_SIZE);

            if (i == 0) {

                for (j = 0; j < L2_SIZE; j++) {

                    p[j].u.leaf = phys_section_unassigned;

                }

            }

        }

        lp = &lp->u.node[(index >> (i * L2_BITS)) & (L2_SIZE - 1)];

    }



    return &lp->u.leaf;

}
