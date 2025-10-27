static void trigger_page_fault(CPUS390XState *env, target_ulong vaddr,

                               uint32_t type, uint64_t asc, int rw)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    int ilen = ILEN_LATER;

    int bits = trans_bits(env, asc);



    /* Code accesses have an undefined ilc.  */

    if (rw == 2) {

        ilen = 2;

    }



    DPRINTF("%s: vaddr=%016" PRIx64 " bits=%d\n", __func__, vaddr, bits);



    stq_phys(cs->as,

             env->psa + offsetof(LowCore, trans_exc_code), vaddr | bits);

    trigger_pgm_exception(env, type, ilen);

}
