static PhysPageDesc *phys_page_find_alloc(target_phys_addr_t index, int alloc)

{

    void **lp, **p;



    p = (void **)l1_phys_map;

#if TARGET_PHYS_ADDR_SPACE_BITS > 32



#if TARGET_PHYS_ADDR_SPACE_BITS > (32 + L1_BITS)

#error unsupported TARGET_PHYS_ADDR_SPACE_BITS

#endif

    lp = p + ((index >> (L1_BITS + L2_BITS)) & (L1_SIZE - 1));

    p = *lp;

    if (!p) {

        /* allocate if not found */

        if (!alloc)

            return NULL;

        p = qemu_vmalloc(sizeof(void *) * L1_SIZE);

        memset(p, 0, sizeof(void *) * L1_SIZE);

        *lp = p;

    }

#endif

    lp = p + ((index >> L2_BITS) & (L1_SIZE - 1));

    p = *lp;

    if (!p) {

        /* allocate if not found */

        if (!alloc)

            return NULL;

        p = qemu_vmalloc(sizeof(PhysPageDesc) * L2_SIZE);

        memset(p, 0, sizeof(PhysPageDesc) * L2_SIZE);

        *lp = p;

    }

    return ((PhysPageDesc *)p) + (index & (L2_SIZE - 1));

}
