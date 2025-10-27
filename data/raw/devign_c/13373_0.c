uint32_t HELPER(mvcos)(CPUS390XState *env, uint64_t dest, uint64_t src,

                       uint64_t len)

{

    const uint8_t psw_key = (env->psw.mask & PSW_MASK_KEY) >> PSW_SHIFT_KEY;

    const uint8_t psw_as = (env->psw.mask & PSW_MASK_ASC) >> PSW_SHIFT_ASC;

    const uint64_t r0 = env->regs[0];

    const uintptr_t ra = GETPC();

    CPUState *cs = CPU(s390_env_get_cpu(env));

    uint8_t dest_key, dest_as, dest_k, dest_a;

    uint8_t src_key, src_as, src_k, src_a;

    uint64_t val;

    int cc = 0;



    HELPER_LOG("%s dest %" PRIx64 ", src %" PRIx64 ", len %" PRIx64 "\n",

               __func__, dest, src, len);



    if (!(env->psw.mask & PSW_MASK_DAT)) {

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_SPECIAL_OP, 6);

    }



    /* OAC (operand access control) for the first operand -> dest */

    val = (r0 & 0xffff0000ULL) >> 16;

    dest_key = (val >> 12) & 0xf;

    dest_as = (val >> 6) & 0x3;

    dest_k = (val >> 1) & 0x1;

    dest_a = val & 0x1;



    /* OAC (operand access control) for the second operand -> src */

    val = (r0 & 0x0000ffffULL);

    src_key = (val >> 12) & 0xf;

    src_as = (val >> 6) & 0x3;

    src_k = (val >> 1) & 0x1;

    src_a = val & 0x1;



    if (!dest_k) {

        dest_key = psw_key;

    }

    if (!src_k) {

        src_key = psw_key;

    }

    if (!dest_a) {

        dest_as = psw_as;

    }

    if (!src_a) {

        src_as = psw_as;

    }



    if (dest_a && dest_as == AS_HOME && (env->psw.mask & PSW_MASK_PSTATE)) {

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_SPECIAL_OP, 6);

    }

    if (!(env->cregs[0] & CR0_SECONDARY) &&

        (dest_as == AS_SECONDARY || src_as == AS_SECONDARY)) {

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_SPECIAL_OP, 6);

    }

    if (!psw_key_valid(env, dest_key) || !psw_key_valid(env, src_key)) {

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_PRIVILEGED, 6);

    }



    len = wrap_length(env, len);

    if (len > 4096) {

        cc = 3;

        len = 4096;

    }



    /* FIXME: AR-mode and proper problem state mode (using PSW keys) missing */

    if (src_as == AS_ACCREG || dest_as == AS_ACCREG ||

        (env->psw.mask & PSW_MASK_PSTATE)) {

        qemu_log_mask(LOG_UNIMP, "%s: AR-mode and PSTATE support missing\n",

                      __func__);

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_ADDRESSING, 6);

    }



    /* FIXME: a) LAP

     *        b) Access using correct keys

     *        c) AR-mode

     */

#ifdef CONFIG_USER_ONLY

    /* psw keys are never valid in user mode, we will never reach this */

    g_assert_not_reached();

#else

    fast_memmove_as(env, dest, src, len, dest_as, src_as, ra);

#endif



    return cc;

}
