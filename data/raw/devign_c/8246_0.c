void helper_movl_sreg_reg (uint32_t sreg, uint32_t reg)

{

	uint32_t srs;

	srs = env->pregs[PR_SRS];

	srs &= 3;

	env->sregs[srs][sreg] = env->regs[reg];



#if !defined(CONFIG_USER_ONLY)

	if (srs == 1 || srs == 2) {

		if (sreg == 6) {

			/* Writes to tlb-hi write to mm_cause as a side 

			   effect.  */

			env->sregs[SFR_RW_MM_TLB_HI] = env->regs[reg];

			env->sregs[SFR_R_MM_CAUSE] = env->regs[reg];

		}

		else if (sreg == 5) {

			uint32_t set;

			uint32_t idx;

			uint32_t lo, hi;

			uint32_t vaddr;

			int tlb_v;



			idx = set = env->sregs[SFR_RW_MM_TLB_SEL];

			set >>= 4;

			set &= 3;



			idx &= 15;

			/* We've just made a write to tlb_lo.  */

			lo = env->sregs[SFR_RW_MM_TLB_LO];

			/* Writes are done via r_mm_cause.  */

			hi = env->sregs[SFR_R_MM_CAUSE];



			vaddr = EXTRACT_FIELD(env->tlbsets[srs-1][set][idx].hi,

					      13, 31);

			vaddr <<= TARGET_PAGE_BITS;

			tlb_v = EXTRACT_FIELD(env->tlbsets[srs-1][set][idx].lo,

					    3, 3);

			env->tlbsets[srs - 1][set][idx].lo = lo;

			env->tlbsets[srs - 1][set][idx].hi = hi;



			D_LOG("tlb flush vaddr=%x v=%d pc=%x\n", 

				  vaddr, tlb_v, env->pc);

			tlb_flush_page(env, vaddr);

		}

	}

#endif

}
