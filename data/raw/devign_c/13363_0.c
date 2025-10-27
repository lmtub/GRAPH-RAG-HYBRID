uint64_t HELPER(lra)(CPUS390XState *env, uint64_t addr)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    uint32_t cc = 0;

    int old_exc = cs->exception_index;

    uint64_t asc = env->psw.mask & PSW_MASK_ASC;

    uint64_t ret;

    int flags;



    /* XXX incomplete - has more corner cases */

    if (!(env->psw.mask & PSW_MASK_64) && (addr >> 32)) {

        program_interrupt(env, PGM_SPECIAL_OP, 2);

    }



    cs->exception_index = old_exc;

    if (mmu_translate(env, addr, 0, asc, &ret, &flags)) {

        cc = 3;

    }

    if (cs->exception_index == EXCP_PGM) {

        ret = env->int_pgm_code | 0x80000000;

    } else {

        ret |= addr & ~TARGET_PAGE_MASK;

    }

    cs->exception_index = old_exc;



    env->cc_op = cc;

    return ret;

}
