int find_itlb_entry(CPUState * env, target_ulong address,

		    int use_asid, int update)

{

    int e, n;



    e = find_tlb_entry(env, address, env->itlb, ITLB_SIZE, use_asid);

    if (e == MMU_DTLB_MULTIPLE)

	e = MMU_ITLB_MULTIPLE;

    else if (e == MMU_DTLB_MISS && update) {

	e = find_tlb_entry(env, address, env->utlb, UTLB_SIZE, use_asid);

	if (e >= 0) {

	    n = itlb_replacement(env);

	    env->itlb[n] = env->utlb[e];

	    e = n;

	} else if (e == MMU_DTLB_MISS)

	    e = MMU_ITLB_MISS;

    } else if (e == MMU_DTLB_MISS)

	e = MMU_ITLB_MISS;

    if (e >= 0)

	update_itlb_use(env, e);

    return e;

}
