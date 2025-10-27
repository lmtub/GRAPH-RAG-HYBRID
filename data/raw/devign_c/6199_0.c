void portio_list_init(PortioList *piolist,

                      const MemoryRegionPortio *callbacks,

                      void *opaque, const char *name)

{

    unsigned n = 0;



    while (callbacks[n].size) {

        ++n;

    }



    piolist->ports = callbacks;

    piolist->nr = 0;

    piolist->regions = g_new0(MemoryRegion *, n);

    piolist->aliases = g_new0(MemoryRegion *, n);

    piolist->address_space = NULL;

    piolist->opaque = opaque;

    piolist->name = name;

}
