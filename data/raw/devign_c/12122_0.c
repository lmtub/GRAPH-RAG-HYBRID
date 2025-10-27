static const ppc_def_t *ppc_find_by_pvr (uint32_t pvr)

{

    int i;



    for (i = 0; i < ARRAY_SIZE(ppc_defs); i++) {

        /* If we have an exact match, we're done */

        if (pvr == ppc_defs[i].pvr) {

            return &ppc_defs[i];

        }

    }



    return NULL;

}
