int cpu_sparc_handle_mmu_fault (CPUState *env, uint32_t address, int rw,

                              int is_user, int is_softmmu)

{

    int exception = 0;

    int access_type, access_perms = 0, access_index = 0;

    uint8_t *pde_ptr;

    uint32_t pde, virt_addr;

    int error_code = 0, is_dirty, prot, ret = 0;

    unsigned long paddr, vaddr, page_offset;



    access_type = env->access_type;

    if (env->user_mode_only) {

        /* user mode only emulation */

        ret = -2;

        goto do_fault;

    }



    virt_addr = address & TARGET_PAGE_MASK;

    if ((env->mmuregs[0] & MMU_E) == 0) { /* MMU disabled */

	paddr = address;

	page_offset = address & (TARGET_PAGE_SIZE - 1);

        prot = PAGE_READ | PAGE_WRITE;

        goto do_mapping;

    }



    /* SPARC reference MMU table walk: Context table->L1->L2->PTE */

    /* Context base + context number */

    pde_ptr = phys_ram_base + (env->mmuregs[1] << 4) + (env->mmuregs[2] << 4);

    env->access_type = ACCESS_MMU;

    pde = ldl_raw(pde_ptr);



    /* Ctx pde */

    switch (pde & PTE_ENTRYTYPE_MASK) {

    case 0: /* Invalid */

        error_code = 1;

        goto do_fault;

    case 2: /* PTE, maybe should not happen? */

    case 3: /* Reserved */

        error_code = 4;

        goto do_fault;

    case 1: /* L1 PDE */

	pde_ptr = phys_ram_base + ((address >> 22) & ~3) + ((pde & ~3) << 4);

	pde = ldl_raw(pde_ptr);



	switch (pde & PTE_ENTRYTYPE_MASK) {

	case 0: /* Invalid */

	    error_code = 1;

	    goto do_fault;

	case 3: /* Reserved */

	    error_code = 4;

	    goto do_fault;

	case 1: /* L2 PDE */

	    pde_ptr = phys_ram_base + ((address & 0xfc0000) >> 16) + ((pde & ~3) << 4);

	    pde = ldl_raw(pde_ptr);



	    switch (pde & PTE_ENTRYTYPE_MASK) {

	    case 0: /* Invalid */

		error_code = 1;

		goto do_fault;

	    case 3: /* Reserved */

		error_code = 4;

		goto do_fault;

	    case 1: /* L3 PDE */

		pde_ptr = phys_ram_base + ((address & 0x3f000) >> 10) + ((pde & ~3) << 4);

		pde = ldl_raw(pde_ptr);



		switch (pde & PTE_ENTRYTYPE_MASK) {

		case 0: /* Invalid */

		    error_code = 1;

		    goto do_fault;

		case 1: /* PDE, should not happen */

		case 3: /* Reserved */

		    error_code = 4;

		    goto do_fault;

		case 2: /* L3 PTE */

		    virt_addr = address & TARGET_PAGE_MASK;

		    page_offset = (address & TARGET_PAGE_MASK) & (TARGET_PAGE_SIZE - 1);

		}

		break;

	    case 2: /* L2 PTE */

		virt_addr = address & ~0x3ffff;

		page_offset = address & 0x3ffff;

	    }

	    break;

	case 2: /* L1 PTE */

	    virt_addr = address & ~0xffffff;

	    page_offset = address & 0xffffff;

	}

    }



    /* update page modified and dirty bits */

    is_dirty = rw && !(pde & PG_MODIFIED_MASK);

    if (!(pde & PG_ACCESSED_MASK) || is_dirty) {

	pde |= PG_ACCESSED_MASK;

	if (is_dirty)

	    pde |= PG_MODIFIED_MASK;

	stl_raw(pde_ptr, pde);

    }



    /* check access */

    access_index = (rw << 2) | ((access_type == ACCESS_CODE)? 2 : 0) | (is_user? 0 : 1);

    access_perms = (pde & PTE_ACCESS_MASK) >> PTE_ACCESS_SHIFT;

    error_code = access_table[access_index][access_perms];

    if (error_code)

	goto do_fault;



    /* the page can be put in the TLB */

    prot = PAGE_READ;

    if (pde & PG_MODIFIED_MASK) {

        /* only set write access if already dirty... otherwise wait

           for dirty access */

	if (rw_table[is_user][access_perms])

	        prot |= PAGE_WRITE;

    }



    /* Even if large ptes, we map only one 4KB page in the cache to

       avoid filling it too fast */

    virt_addr = address & TARGET_PAGE_MASK;

    paddr = ((pde & PTE_ADDR_MASK) << 4) + page_offset;



 do_mapping:

    env->access_type = access_type;

    vaddr = virt_addr + ((address & TARGET_PAGE_MASK) & (TARGET_PAGE_SIZE - 1));



    ret = tlb_set_page(env, vaddr, paddr, prot, is_user, is_softmmu);

    return ret;



 do_fault:

    env->access_type = access_type;

    if (env->mmuregs[3]) /* Fault status register */

	env->mmuregs[3] = 1; /* overflow (not read before another fault) */

    env->mmuregs[3] |= (access_index << 5) | (error_code << 2) | 2;

    env->mmuregs[4] = address; /* Fault address register */



    if (env->mmuregs[0] & MMU_NF) // No fault

	return 0;



    env->exception_index = exception;

    env->error_code = error_code;

    return error_code;

}
