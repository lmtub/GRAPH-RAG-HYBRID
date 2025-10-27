void helper_tlb_update(uint32_t T0)

{

#if !defined(CONFIG_USER_ONLY)

	uint32_t vaddr;

	uint32_t srs = env->pregs[PR_SRS];



	if (srs != 1 && srs != 2)

		return;



	vaddr = cris_mmu_tlb_latest_update(env, T0);

	D(printf("flush old_vaddr=%x vaddr=%x T0=%x\n", vaddr, 

		 env->sregs[SFR_R_MM_CAUSE] & TARGET_PAGE_MASK, T0));

	tlb_flush_page(env, vaddr);

#endif

}
