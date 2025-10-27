static int get_physical_address(CPUState *env, target_phys_addr_t *physical,

                                int *prot, int *access_index,

                                target_ulong address, int rw, int mmu_idx)

{

    int access_perms = 0;

    target_phys_addr_t pde_ptr;

    uint32_t pde;

    int error_code = 0, is_dirty, is_user;

    unsigned long page_offset;



    is_user = mmu_idx == MMU_USER_IDX;



    if ((env->mmuregs[0] & MMU_E) == 0) { /* MMU disabled */

        // Boot mode: instruction fetches are taken from PROM

        if (rw == 2 && (env->mmuregs[0] & env->def->mmu_bm)) {

            *physical = env->prom_addr | (address & 0x7ffffULL);

            *prot = PAGE_READ | PAGE_EXEC;

            return 0;

        }

        *physical = address;

        *prot = PAGE_READ | PAGE_WRITE | PAGE_EXEC;

        return 0;

    }



    *access_index = ((rw & 1) << 2) | (rw & 2) | (is_user? 0 : 1);

    *physical = 0xffffffffffff0000ULL;



    /* SPARC reference MMU table walk: Context table->L1->L2->PTE */

    /* Context base + context number */

    pde_ptr = (env->mmuregs[1] << 4) + (env->mmuregs[2] << 2);

    pde = ldl_phys(pde_ptr);



    /* Ctx pde */

    switch (pde & PTE_ENTRYTYPE_MASK) {

    default:

    case 0: /* Invalid */

        return 1 << 2;

    case 2: /* L0 PTE, maybe should not happen? */

    case 3: /* Reserved */

        return 4 << 2;

    case 1: /* L0 PDE */

        pde_ptr = ((address >> 22) & ~3) + ((pde & ~3) << 4);

        pde = ldl_phys(pde_ptr);



        switch (pde & PTE_ENTRYTYPE_MASK) {

        default:

        case 0: /* Invalid */

            return (1 << 8) | (1 << 2);

        case 3: /* Reserved */

            return (1 << 8) | (4 << 2);

        case 1: /* L1 PDE */

            pde_ptr = ((address & 0xfc0000) >> 16) + ((pde & ~3) << 4);

            pde = ldl_phys(pde_ptr);



            switch (pde & PTE_ENTRYTYPE_MASK) {

            default:

            case 0: /* Invalid */

                return (2 << 8) | (1 << 2);

            case 3: /* Reserved */

                return (2 << 8) | (4 << 2);

            case 1: /* L2 PDE */

                pde_ptr = ((address & 0x3f000) >> 10) + ((pde & ~3) << 4);

                pde = ldl_phys(pde_ptr);



                switch (pde & PTE_ENTRYTYPE_MASK) {

                default:

                case 0: /* Invalid */

                    return (3 << 8) | (1 << 2);

                case 1: /* PDE, should not happen */

                case 3: /* Reserved */

                    return (3 << 8) | (4 << 2);

                case 2: /* L3 PTE */

                    page_offset = (address & TARGET_PAGE_MASK) &

                        (TARGET_PAGE_SIZE - 1);

                }

                break;

            case 2: /* L2 PTE */

                page_offset = address & 0x3ffff;

            }

            break;

        case 2: /* L1 PTE */

            page_offset = address & 0xffffff;

        }

    }



    /* update page modified and dirty bits */

    is_dirty = (rw & 1) && !(pde & PG_MODIFIED_MASK);

    if (!(pde & PG_ACCESSED_MASK) || is_dirty) {

        pde |= PG_ACCESSED_MASK;

        if (is_dirty)

            pde |= PG_MODIFIED_MASK;

        stl_phys_notdirty(pde_ptr, pde);

    }

    /* check access */

    access_perms = (pde & PTE_ACCESS_MASK) >> PTE_ACCESS_SHIFT;

    error_code = access_table[*access_index][access_perms];

    if (error_code && !((env->mmuregs[0] & MMU_NF) && is_user))

        return error_code;



    /* the page can be put in the TLB */

    *prot = perm_table[is_user][access_perms];

    if (!(pde & PG_MODIFIED_MASK)) {

        /* only set write access if already dirty... otherwise wait

           for dirty access */

        *prot &= ~PAGE_WRITE;

    }



    /* Even if large ptes, we map only one 4KB page in the cache to

       avoid filling it too fast */

    *physical = ((target_phys_addr_t)(pde & PTE_ADDR_MASK) << 4) + page_offset;

    return error_code;

}
