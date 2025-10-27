static int xen_add_to_physmap(XenIOState *state,

                              hwaddr start_addr,

                              ram_addr_t size,

                              MemoryRegion *mr,

                              hwaddr offset_within_region)

{

    unsigned long i = 0;

    int rc = 0;

    XenPhysmap *physmap = NULL;

    hwaddr pfn, start_gpfn;

    hwaddr phys_offset = memory_region_get_ram_addr(mr);

    char path[80], value[17];

    const char *mr_name;



    if (get_physmapping(state, start_addr, size)) {

        return 0;

    }

    if (size <= 0) {

        return -1;

    }



    /* Xen can only handle a single dirty log region for now and we want

     * the linear framebuffer to be that region.

     * Avoid tracking any regions that is not videoram and avoid tracking

     * the legacy vga region. */

    if (mr == framebuffer && start_addr > 0xbffff) {

        goto go_physmap;

    }

    return -1;



go_physmap:

    DPRINTF("mapping vram to %"HWADDR_PRIx" - %"HWADDR_PRIx"\n",

            start_addr, start_addr + size);



    pfn = phys_offset >> TARGET_PAGE_BITS;

    start_gpfn = start_addr >> TARGET_PAGE_BITS;

    for (i = 0; i < size >> TARGET_PAGE_BITS; i++) {

        unsigned long idx = pfn + i;

        xen_pfn_t gpfn = start_gpfn + i;



        rc = xc_domain_add_to_physmap(xen_xc, xen_domid, XENMAPSPACE_gmfn, idx, gpfn);

        if (rc) {

            DPRINTF("add_to_physmap MFN %"PRI_xen_pfn" to PFN %"

                    PRI_xen_pfn" failed: %d (errno: %d)\n", idx, gpfn, rc, errno);

            return -rc;

        }

    }



    mr_name = memory_region_name(mr);



    physmap = g_malloc(sizeof (XenPhysmap));



    physmap->start_addr = start_addr;

    physmap->size = size;

    physmap->name = mr_name;

    physmap->phys_offset = phys_offset;



    QLIST_INSERT_HEAD(&state->physmap, physmap, list);



    xc_domain_pin_memory_cacheattr(xen_xc, xen_domid,

                                   start_addr >> TARGET_PAGE_BITS,

                                   (start_addr + size - 1) >> TARGET_PAGE_BITS,

                                   XEN_DOMCTL_MEM_CACHEATTR_WB);



    snprintf(path, sizeof(path),

            "/local/domain/0/device-model/%d/physmap/%"PRIx64"/start_addr",

            xen_domid, (uint64_t)phys_offset);

    snprintf(value, sizeof(value), "%"PRIx64, (uint64_t)start_addr);

    if (!xs_write(state->xenstore, 0, path, value, strlen(value))) {

        return -1;

    }

    snprintf(path, sizeof(path),

            "/local/domain/0/device-model/%d/physmap/%"PRIx64"/size",

            xen_domid, (uint64_t)phys_offset);

    snprintf(value, sizeof(value), "%"PRIx64, (uint64_t)size);

    if (!xs_write(state->xenstore, 0, path, value, strlen(value))) {

        return -1;

    }

    if (mr_name) {

        snprintf(path, sizeof(path),

                "/local/domain/0/device-model/%d/physmap/%"PRIx64"/name",

                xen_domid, (uint64_t)phys_offset);

        if (!xs_write(state->xenstore, 0, path, mr_name, strlen(mr_name))) {

            return -1;

        }

    }



    return 0;

}
