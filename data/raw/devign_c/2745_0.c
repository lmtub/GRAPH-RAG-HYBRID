target_ulong cpu_get_phys_page_debug(CPUState *env, target_ulong addr)

{

    uint8_t *pde_ptr, *pte_ptr;

    uint32_t pde, pte, paddr, page_offset, page_size;



    if (!(env->cr[0] & CR0_PG_MASK)) {

        pte = addr;

        page_size = 4096;

    } else {

        /* page directory entry */

        pde_ptr = phys_ram_base + 

            (((env->cr[3] & ~0xfff) + ((addr >> 20) & ~3)) & a20_mask);

        pde = ldl_raw(pde_ptr);

        if (!(pde & PG_PRESENT_MASK)) 

            return -1;

        if ((pde & PG_PSE_MASK) && (env->cr[4] & CR4_PSE_MASK)) {

            pte = pde & ~0x003ff000; /* align to 4MB */

            page_size = 4096 * 1024;

        } else {

            /* page directory entry */

            pte_ptr = phys_ram_base + 

                (((pde & ~0xfff) + ((addr >> 10) & 0xffc)) & a20_mask);

            pte = ldl_raw(pte_ptr);

            if (!(pte & PG_PRESENT_MASK))

                return -1;

            page_size = 4096;

        }

    }

    pte = pte & a20_mask;

    page_offset = (addr & TARGET_PAGE_MASK) & (page_size - 1);

    paddr = (pte & TARGET_PAGE_MASK) + page_offset;

    return paddr;

}
