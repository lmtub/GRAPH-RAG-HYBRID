int cpu_sh4_handle_mmu_fault(CPUState * env, target_ulong address, int rw,

			     int mmu_idx, int is_softmmu)

{

    target_ulong physical;

    int prot, ret, access_type;



    access_type = ACCESS_INT;

    ret =

	get_physical_address(env, &physical, &prot, address, rw,

			     access_type);



    if (ret != MMU_OK) {

	env->tea = address;

	switch (ret) {

	case MMU_ITLB_MISS:

	case MMU_DTLB_MISS_READ:

	    env->exception_index = 0x040;

	    break;

	case MMU_DTLB_MULTIPLE:

	case MMU_ITLB_MULTIPLE:

	    env->exception_index = 0x140;

	    break;

	case MMU_ITLB_VIOLATION:

	    env->exception_index = 0x0a0;

	    break;

	case MMU_DTLB_MISS_WRITE:

	    env->exception_index = 0x060;

	    break;

	case MMU_DTLB_INITIAL_WRITE:

	    env->exception_index = 0x080;

	    break;

	case MMU_DTLB_VIOLATION_READ:

	    env->exception_index = 0x0a0;

	    break;

	case MMU_DTLB_VIOLATION_WRITE:

	    env->exception_index = 0x0c0;

	    break;

	case MMU_IADDR_ERROR:

	case MMU_DADDR_ERROR_READ:

	    env->exception_index = 0x0c0;

	    break;

	case MMU_DADDR_ERROR_WRITE:

	    env->exception_index = 0x100;

	    break;

	default:

	    assert(0);

	}

	return 1;

    }



    address &= TARGET_PAGE_MASK;

    physical &= TARGET_PAGE_MASK;



    return tlb_set_page(env, address, physical, prot, mmu_idx, is_softmmu);

}
