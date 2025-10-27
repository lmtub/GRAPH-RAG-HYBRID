int cpu_x86_handle_mmu_fault(CPUX86State *env, target_ulong addr, 

                             int is_write1, int is_user, int is_softmmu)

{

    uint64_t ptep, pte;

    uint32_t pdpe_addr, pde_addr, pte_addr;

    int error_code, is_dirty, prot, page_size, ret, is_write;

    unsigned long paddr, page_offset;

    target_ulong vaddr, virt_addr;

    

#if defined(DEBUG_MMU)

    printf("MMU fault: addr=" TARGET_FMT_lx " w=%d u=%d eip=" TARGET_FMT_lx "\n", 

           addr, is_write1, is_user, env->eip);

#endif

    is_write = is_write1 & 1;

    

    if (!(env->cr[0] & CR0_PG_MASK)) {

        pte = addr;

        virt_addr = addr & TARGET_PAGE_MASK;

        prot = PAGE_READ | PAGE_WRITE | PAGE_EXEC;

        page_size = 4096;

        goto do_mapping;

    }



    if (env->cr[4] & CR4_PAE_MASK) {

        uint64_t pde, pdpe;



        /* XXX: we only use 32 bit physical addresses */

#ifdef TARGET_X86_64

        if (env->hflags & HF_LMA_MASK) {

            uint32_t pml4e_addr;

            uint64_t pml4e;

            int32_t sext;



            /* test virtual address sign extension */

            sext = (int64_t)addr >> 47;

            if (sext != 0 && sext != -1) {

                error_code = 0;

                goto do_fault;

            }

            

            pml4e_addr = ((env->cr[3] & ~0xfff) + (((addr >> 39) & 0x1ff) << 3)) & 

                env->a20_mask;

            pml4e = ldq_phys(pml4e_addr);

            if (!(pml4e & PG_PRESENT_MASK)) {

                error_code = 0;

                goto do_fault;

            }

            if (!(env->efer & MSR_EFER_NXE) && (pml4e & PG_NX_MASK)) {

                error_code = PG_ERROR_RSVD_MASK;

                goto do_fault;

            }

            if (!(pml4e & PG_ACCESSED_MASK)) {

                pml4e |= PG_ACCESSED_MASK;

                stl_phys_notdirty(pml4e_addr, pml4e);

            }

            ptep = pml4e ^ PG_NX_MASK;

            pdpe_addr = ((pml4e & PHYS_ADDR_MASK) + (((addr >> 30) & 0x1ff) << 3)) & 

                env->a20_mask;

            pdpe = ldq_phys(pdpe_addr);

            if (!(pdpe & PG_PRESENT_MASK)) {

                error_code = 0;

                goto do_fault;

            }

            if (!(env->efer & MSR_EFER_NXE) && (pdpe & PG_NX_MASK)) {

                error_code = PG_ERROR_RSVD_MASK;

                goto do_fault;

            }

            ptep &= pdpe ^ PG_NX_MASK;

            if (!(pdpe & PG_ACCESSED_MASK)) {

                pdpe |= PG_ACCESSED_MASK;

                stl_phys_notdirty(pdpe_addr, pdpe);

            }

        } else

#endif

        {

            /* XXX: load them when cr3 is loaded ? */

            pdpe_addr = ((env->cr[3] & ~0x1f) + ((addr >> 30) << 3)) & 

                env->a20_mask;

            pdpe = ldq_phys(pdpe_addr);

            if (!(pdpe & PG_PRESENT_MASK)) {

                error_code = 0;

                goto do_fault;

            }

            ptep = PG_NX_MASK | PG_USER_MASK | PG_RW_MASK;

        }



        pde_addr = ((pdpe & PHYS_ADDR_MASK) + (((addr >> 21) & 0x1ff) << 3)) &

            env->a20_mask;

        pde = ldq_phys(pde_addr);

        if (!(pde & PG_PRESENT_MASK)) {

            error_code = 0;

            goto do_fault;

        }

        if (!(env->efer & MSR_EFER_NXE) && (pde & PG_NX_MASK)) {

            error_code = PG_ERROR_RSVD_MASK;

            goto do_fault;

        }

        ptep &= pde ^ PG_NX_MASK;

        if (pde & PG_PSE_MASK) {

            /* 2 MB page */

            page_size = 2048 * 1024;

            ptep ^= PG_NX_MASK;

            if ((ptep & PG_NX_MASK) && is_write1 == 2)

                goto do_fault_protect;

            if (is_user) {

                if (!(ptep & PG_USER_MASK))

                    goto do_fault_protect;

                if (is_write && !(ptep & PG_RW_MASK))

                    goto do_fault_protect;

            } else {

                if ((env->cr[0] & CR0_WP_MASK) && 

                    is_write && !(ptep & PG_RW_MASK)) 

                    goto do_fault_protect;

            }

            is_dirty = is_write && !(pde & PG_DIRTY_MASK);

            if (!(pde & PG_ACCESSED_MASK) || is_dirty) {

                pde |= PG_ACCESSED_MASK;

                if (is_dirty)

                    pde |= PG_DIRTY_MASK;

                stl_phys_notdirty(pde_addr, pde);

            }

            /* align to page_size */

            pte = pde & ((PHYS_ADDR_MASK & ~(page_size - 1)) | 0xfff); 

            virt_addr = addr & ~(page_size - 1);

        } else {

            /* 4 KB page */

            if (!(pde & PG_ACCESSED_MASK)) {

                pde |= PG_ACCESSED_MASK;

                stl_phys_notdirty(pde_addr, pde);

            }

            pte_addr = ((pde & PHYS_ADDR_MASK) + (((addr >> 12) & 0x1ff) << 3)) &

                env->a20_mask;

            pte = ldq_phys(pte_addr);

            if (!(pte & PG_PRESENT_MASK)) {

                error_code = 0;

                goto do_fault;

            }

            if (!(env->efer & MSR_EFER_NXE) && (pte & PG_NX_MASK)) {

                error_code = PG_ERROR_RSVD_MASK;

                goto do_fault;

            }

            /* combine pde and pte nx, user and rw protections */

            ptep &= pte ^ PG_NX_MASK;

            ptep ^= PG_NX_MASK;

            if ((ptep & PG_NX_MASK) && is_write1 == 2)

                goto do_fault_protect; 

            if (is_user) {

                if (!(ptep & PG_USER_MASK))

                    goto do_fault_protect;

                if (is_write && !(ptep & PG_RW_MASK))

                    goto do_fault_protect;

            } else {

                if ((env->cr[0] & CR0_WP_MASK) &&

                    is_write && !(ptep & PG_RW_MASK)) 

                    goto do_fault_protect;

            }

            is_dirty = is_write && !(pte & PG_DIRTY_MASK);

            if (!(pte & PG_ACCESSED_MASK) || is_dirty) {

                pte |= PG_ACCESSED_MASK;

                if (is_dirty)

                    pte |= PG_DIRTY_MASK;

                stl_phys_notdirty(pte_addr, pte);

            }

            page_size = 4096;

            virt_addr = addr & ~0xfff;

            pte = pte & (PHYS_ADDR_MASK | 0xfff);

        }

    } else {

        uint32_t pde;



        /* page directory entry */

        pde_addr = ((env->cr[3] & ~0xfff) + ((addr >> 20) & ~3)) & 

            env->a20_mask;

        pde = ldl_phys(pde_addr);

        if (!(pde & PG_PRESENT_MASK)) {

            error_code = 0;

            goto do_fault;

        }

        /* if PSE bit is set, then we use a 4MB page */

        if ((pde & PG_PSE_MASK) && (env->cr[4] & CR4_PSE_MASK)) {

            page_size = 4096 * 1024;

            if (is_user) {

                if (!(pde & PG_USER_MASK))

                    goto do_fault_protect;

                if (is_write && !(pde & PG_RW_MASK))

                    goto do_fault_protect;

            } else {

                if ((env->cr[0] & CR0_WP_MASK) && 

                    is_write && !(pde & PG_RW_MASK)) 

                    goto do_fault_protect;

            }

            is_dirty = is_write && !(pde & PG_DIRTY_MASK);

            if (!(pde & PG_ACCESSED_MASK) || is_dirty) {

                pde |= PG_ACCESSED_MASK;

                if (is_dirty)

                    pde |= PG_DIRTY_MASK;

                stl_phys_notdirty(pde_addr, pde);

            }

        

            pte = pde & ~( (page_size - 1) & ~0xfff); /* align to page_size */

            ptep = pte;

            virt_addr = addr & ~(page_size - 1);

        } else {

            if (!(pde & PG_ACCESSED_MASK)) {

                pde |= PG_ACCESSED_MASK;

                stl_phys_notdirty(pde_addr, pde);

            }



            /* page directory entry */

            pte_addr = ((pde & ~0xfff) + ((addr >> 10) & 0xffc)) & 

                env->a20_mask;

            pte = ldl_phys(pte_addr);

            if (!(pte & PG_PRESENT_MASK)) {

                error_code = 0;

                goto do_fault;

            }

            /* combine pde and pte user and rw protections */

            ptep = pte & pde;

            if (is_user) {

                if (!(ptep & PG_USER_MASK))

                    goto do_fault_protect;

                if (is_write && !(ptep & PG_RW_MASK))

                    goto do_fault_protect;

            } else {

                if ((env->cr[0] & CR0_WP_MASK) &&

                    is_write && !(ptep & PG_RW_MASK)) 

                    goto do_fault_protect;

            }

            is_dirty = is_write && !(pte & PG_DIRTY_MASK);

            if (!(pte & PG_ACCESSED_MASK) || is_dirty) {

                pte |= PG_ACCESSED_MASK;

                if (is_dirty)

                    pte |= PG_DIRTY_MASK;

                stl_phys_notdirty(pte_addr, pte);

            }

            page_size = 4096;

            virt_addr = addr & ~0xfff;

        }

    }

    /* the page can be put in the TLB */

    prot = PAGE_READ;

    if (!(ptep & PG_NX_MASK))

        prot |= PAGE_EXEC;

    if (pte & PG_DIRTY_MASK) {

        /* only set write access if already dirty... otherwise wait

           for dirty access */

        if (is_user) {

            if (ptep & PG_RW_MASK)

                prot |= PAGE_WRITE;

        } else {

            if (!(env->cr[0] & CR0_WP_MASK) ||

                (ptep & PG_RW_MASK))

                prot |= PAGE_WRITE;

        }

    }

 do_mapping:

    pte = pte & env->a20_mask;



    /* Even if 4MB pages, we map only one 4KB page in the cache to

       avoid filling it too fast */

    page_offset = (addr & TARGET_PAGE_MASK) & (page_size - 1);

    paddr = (pte & TARGET_PAGE_MASK) + page_offset;

    vaddr = virt_addr + page_offset;

    

    ret = tlb_set_page_exec(env, vaddr, paddr, prot, is_user, is_softmmu);

    return ret;

 do_fault_protect:

    error_code = PG_ERROR_P_MASK;

 do_fault:

    env->cr[2] = addr;

    error_code |= (is_write << PG_ERROR_W_BIT);

    if (is_user)

        error_code |= PG_ERROR_U_MASK;

    if (is_write1 == 2 && 

        (env->efer & MSR_EFER_NXE) && 

        (env->cr[4] & CR4_PAE_MASK))

        error_code |= PG_ERROR_I_D_MASK;

    env->error_code = error_code;

    return 1;

}
