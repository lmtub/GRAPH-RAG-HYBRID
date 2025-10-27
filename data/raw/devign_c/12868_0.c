void restore_state_to_opc(CPUARMState *env, TranslationBlock *tb, int pc_pos)

{

    if (is_a64(env)) {

        env->pc = tcg_ctx.gen_opc_pc[pc_pos];

    } else {

        env->regs[15] = tcg_ctx.gen_opc_pc[pc_pos];

    }

    env->condexec_bits = gen_opc_condexec_bits[pc_pos];

}
