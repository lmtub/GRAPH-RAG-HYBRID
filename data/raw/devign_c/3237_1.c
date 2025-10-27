int ppc_find_by_pvr (uint32_t pvr, ppc_def_t **def)

{

    int i, ret;



    ret = -1;

    *def = NULL;

    for (i = 0; ppc_defs[i].name != NULL; i++) {

        if ((pvr & ppc_defs[i].pvr_mask) ==

            (ppc_defs[i].pvr & ppc_defs[i].pvr_mask)) {

            *def = &ppc_defs[i];

            ret = 0;

            break;

        }

    }



    return ret;

}
