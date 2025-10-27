static void xen_log_stop(MemoryListener *listener, MemoryRegionSection *section)

{

    XenIOState *state = container_of(listener, XenIOState, memory_listener);



    state->log_for_dirtybit = NULL;

    /* Disable dirty bit tracking */

    xc_hvm_track_dirty_vram(xen_xc, xen_domid, 0, 0, NULL);

}
