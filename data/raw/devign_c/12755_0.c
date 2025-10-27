hwaddr x86_cpu_get_phys_page_debug(CPUState *cs, vaddr addr)

{

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;

    target_ulong pde_addr, pte_addr;

    uint64_t pte;

    hwaddr paddr;

    uint32_t page_offset;

    int page_size;



    if (env->cr[4] & CR4_PAE_MASK) {

        target_ulong pdpe_addr;

        uint64_t pde, pdpe;



#ifdef TARGET_X86_64

        if (env->hflags & HF_LMA_MASK) {

            uint64_t pml4e_addr, pml4e;

            int32_t sext;



            /* test virtual address sign extension */

            sext = (int64_t)addr >> 47;

            if (sext != 0 && sext != -1)

                return -1;



            pml4e_addr = ((env->cr[3] & ~0xfff) + (((addr >> 39) & 0x1ff) << 3)) &

                env->a20_mask;

            pml4e = ldq_phys(pml4e_addr);

            if (!(pml4e & PG_PRESENT_MASK))

                return -1;



            pdpe_addr = ((pml4e & ~0xfff & ~(PG_NX_MASK | PG_HI_USER_MASK)) +

                         (((addr >> 30) & 0x1ff) << 3)) & env->a20_mask;

            pdpe = ldq_phys(pdpe_addr);

            if (!(pdpe & PG_PRESENT_MASK))

                return -1;

        } else

#endif

        {

            pdpe_addr = ((env->cr[3] & ~0x1f) + ((addr >> 27) & 0x18)) &

                env->a20_mask;

            pdpe = ldq_phys(pdpe_addr);

            if (!(pdpe & PG_PRESENT_MASK))

                return -1;

        }



        pde_addr = ((pdpe & ~0xfff & ~(PG_NX_MASK | PG_HI_USER_MASK)) +

                    (((addr >> 21) & 0x1ff) << 3)) & env->a20_mask;

        pde = ldq_phys(pde_addr);

        if (!(pde & PG_PRESENT_MASK)) {

            return -1;

        }

        if (pde & PG_PSE_MASK) {

            /* 2 MB page */

            page_size = 2048 * 1024;

            pte = pde & ~( (page_size - 1) & ~0xfff); /* align to page_size */

        } else {

            /* 4 KB page */

            pte_addr = ((pde & ~0xfff & ~(PG_NX_MASK | PG_HI_USER_MASK)) +

                        (((addr >> 12) & 0x1ff) << 3)) & env->a20_mask;

            page_size = 4096;

            pte = ldq_phys(pte_addr);

        }

        pte &= ~(PG_NX_MASK | PG_HI_USER_MASK);

        if (!(pte & PG_PRESENT_MASK))

            return -1;

    } else {

        uint32_t pde;



        if (!(env->cr[0] & CR0_PG_MASK)) {

            pte = addr;

            page_size = 4096;

        } else {

            /* page directory entry */

            pde_addr = ((env->cr[3] & ~0xfff) + ((addr >> 20) & 0xffc)) & env->a20_mask;

            pde = ldl_phys(pde_addr);

            if (!(pde & PG_PRESENT_MASK))

                return -1;

            if ((pde & PG_PSE_MASK) && (env->cr[4] & CR4_PSE_MASK)) {

                pte = pde & ~0x003ff000; /* align to 4MB */

                page_size = 4096 * 1024;

            } else {

                /* page directory entry */

                pte_addr = ((pde & ~0xfff) + ((addr >> 10) & 0xffc)) & env->a20_mask;

                pte = ldl_phys(pte_addr);

                if (!(pte & PG_PRESENT_MASK))

                    return -1;

                page_size = 4096;

            }

        }

        pte = pte & env->a20_mask;

    }



    page_offset = (addr & TARGET_PAGE_MASK) & (page_size - 1);

    paddr = (pte & TARGET_PAGE_MASK) + page_offset;

    return paddr;

}
