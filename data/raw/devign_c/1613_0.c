static void xen_io_del(MemoryListener *listener,

                       MemoryRegionSection *section)

{

    XenIOState *state = container_of(listener, XenIOState, io_listener);



    xen_unmap_io_section(xen_xc, xen_domid, state->ioservid, section);



    memory_region_unref(section->mr);

}
