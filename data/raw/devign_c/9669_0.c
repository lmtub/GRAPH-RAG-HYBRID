int get_physical_address(CPUState * env, target_ulong * physical,

			 int *prot, target_ulong address,

			 int rw, int access_type)

{

    /* P1, P2 and P4 areas do not use translation */

    if ((address >= 0x80000000 && address < 0xc0000000) ||

	address >= 0xe0000000) {

	if (!(env->sr & SR_MD)

	    && (address < 0xe0000000 || address > 0xe4000000)) {

	    /* Unauthorized access in user mode (only store queues are available) */

	    fprintf(stderr, "Unauthorized access\n");

	    return (rw & PAGE_WRITE) ? MMU_DTLB_MISS_WRITE :

		MMU_DTLB_MISS_READ;

	}

	/* Mask upper 3 bits */

	*physical = address & 0x1FFFFFFF;

	*prot = PAGE_READ | PAGE_WRITE;

	return MMU_OK;

    }



    /* If MMU is disabled, return the corresponding physical page */

    if (!env->mmucr & MMUCR_AT) {

	*physical = address & 0x1FFFFFFF;

	*prot = PAGE_READ | PAGE_WRITE;

	return MMU_OK;

    }



    /* We need to resort to the MMU */

    return get_mmu_address(env, physical, prot, address, rw, access_type);

}
