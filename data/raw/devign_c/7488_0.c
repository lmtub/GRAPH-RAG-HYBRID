static void xen_io_add(MemoryListener *listener,

                       MemoryRegionSection *section)

{

    XenIOState *state = container_of(listener, XenIOState, io_listener);



    memory_region_ref(section->mr);



    xen_map_io_section(xen_xc, xen_domid, state->ioservid, section);

}
