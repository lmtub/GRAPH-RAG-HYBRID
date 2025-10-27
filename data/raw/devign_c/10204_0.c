static int xen_remove_from_physmap(XenIOState *state,

                                   hwaddr start_addr,

                                   ram_addr_t size)

{

    unsigned long i = 0;

    int rc = 0;

    XenPhysmap *physmap = NULL;

    hwaddr phys_offset = 0;



    physmap = get_physmapping(state, start_addr, size);

    if (physmap == NULL) {

        return -1;

    }



    phys_offset = physmap->phys_offset;

    size = physmap->size;



    DPRINTF("unmapping vram to %"HWADDR_PRIx" - %"HWADDR_PRIx", at "

            "%"HWADDR_PRIx"\n", start_addr, start_addr + size, phys_offset);



    size >>= TARGET_PAGE_BITS;

    start_addr >>= TARGET_PAGE_BITS;

    phys_offset >>= TARGET_PAGE_BITS;

    for (i = 0; i < size; i++) {

        unsigned long idx = start_addr + i;

        xen_pfn_t gpfn = phys_offset + i;



        rc = xc_domain_add_to_physmap(xen_xc, xen_domid, XENMAPSPACE_gmfn, idx, gpfn);

        if (rc) {

            fprintf(stderr, "add_to_physmap MFN %"PRI_xen_pfn" to PFN %"

                    PRI_xen_pfn" failed: %d\n", idx, gpfn, rc);

            return -rc;

        }

    }



    QLIST_REMOVE(physmap, list);

    if (state->log_for_dirtybit == physmap) {

        state->log_for_dirtybit = NULL;

    }

    g_free(physmap);



    return 0;

}
