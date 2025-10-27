static void destroy_l2_mapping(PhysPageEntry *lp, unsigned level)

{

    unsigned i;

    PhysPageEntry *p = lp->u.node;



    if (!p) {

        return;

    }



    for (i = 0; i < L2_SIZE; ++i) {

        if (level > 0) {

            destroy_l2_mapping(&p[i], level - 1);

        } else {

            destroy_page_desc(p[i].u.leaf);

        }

    }

    g_free(p);

    lp->u.node = NULL;

}
