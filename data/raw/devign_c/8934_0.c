void arm_cpu_do_unaligned_access(CPUState *cs, vaddr vaddr, int is_write,

                                 int is_user, uintptr_t retaddr)

{

    ARMCPU *cpu = ARM_CPU(cs);

    CPUARMState *env = &cpu->env;

    int target_el;

    bool same_el;



    if (retaddr) {

        /* now we have a real cpu fault */

        cpu_restore_state(cs, retaddr);

    }



    target_el = exception_target_el(env);

    same_el = (arm_current_el(env) == target_el);



    env->exception.vaddress = vaddr;



    /* the DFSR for an alignment fault depends on whether we're using

     * the LPAE long descriptor format, or the short descriptor format

     */

    if (arm_regime_using_lpae_format(env, cpu_mmu_index(env, false))) {

        env->exception.fsr = 0x21;

    } else {

        env->exception.fsr = 0x1;

    }



    if (is_write == 1 && arm_feature(env, ARM_FEATURE_V6)) {

        env->exception.fsr |= (1 << 11);

    }



    raise_exception(env, EXCP_DATA_ABORT,

                    syn_data_abort(same_el, 0, 0, 0, is_write == 1, 0x21),

                    target_el);

}
