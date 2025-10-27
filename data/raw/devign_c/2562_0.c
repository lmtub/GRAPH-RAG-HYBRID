void portio_list_del(PortioList *piolist)

{

    MemoryRegion *mr, *alias;

    unsigned i;



    for (i = 0; i < piolist->nr; ++i) {

        mr = piolist->regions[i];

        alias = piolist->aliases[i];

        memory_region_del_subregion(piolist->address_space, alias);

        memory_region_destroy(alias);

        memory_region_destroy(mr);

        g_free((MemoryRegionOps *)mr->ops);

        g_free(mr);

        g_free(alias);

        piolist->regions[i] = NULL;

        piolist->aliases[i] = NULL;

    }

}
