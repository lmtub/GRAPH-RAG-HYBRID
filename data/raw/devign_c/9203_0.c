static int aarch64_tr_init_disas_context(DisasContextBase *dcbase,

                                         CPUState *cpu, int max_insns)

{

    DisasContext *dc = container_of(dcbase, DisasContext, base);

    CPUARMState *env = cpu->env_ptr;

    ARMCPU *arm_cpu = arm_env_get_cpu(env);



    dc->pc = dc->base.pc_first;

    dc->condjmp = 0;



    dc->aarch64 = 1;

    /* If we are coming from secure EL0 in a system with a 32-bit EL3, then

     * there is no secure EL1, so we route exceptions to EL3.

     */

    dc->secure_routed_to_el3 = arm_feature(env, ARM_FEATURE_EL3) &&

                               !arm_el_is_aa64(env, 3);

    dc->thumb = 0;

    dc->sctlr_b = 0;

    dc->be_data = ARM_TBFLAG_BE_DATA(dc->base.tb->flags) ? MO_BE : MO_LE;

    dc->condexec_mask = 0;

    dc->condexec_cond = 0;

    dc->mmu_idx = core_to_arm_mmu_idx(env, ARM_TBFLAG_MMUIDX(dc->base.tb->flags));

    dc->tbi0 = ARM_TBFLAG_TBI0(dc->base.tb->flags);

    dc->tbi1 = ARM_TBFLAG_TBI1(dc->base.tb->flags);

    dc->current_el = arm_mmu_idx_to_el(dc->mmu_idx);

#if !defined(CONFIG_USER_ONLY)

    dc->user = (dc->current_el == 0);

#endif

    dc->fp_excp_el = ARM_TBFLAG_FPEXC_EL(dc->base.tb->flags);

    dc->vec_len = 0;

    dc->vec_stride = 0;

    dc->cp_regs = arm_cpu->cp_regs;

    dc->features = env->features;



    /* Single step state. The code-generation logic here is:

     *  SS_ACTIVE == 0:

     *   generate code with no special handling for single-stepping (except

     *   that anything that can make us go to SS_ACTIVE == 1 must end the TB;

     *   this happens anyway because those changes are all system register or

     *   PSTATE writes).

     *  SS_ACTIVE == 1, PSTATE.SS == 1: (active-not-pending)

     *   emit code for one insn

     *   emit code to clear PSTATE.SS

     *   emit code to generate software step exception for completed step

     *   end TB (as usual for having generated an exception)

     *  SS_ACTIVE == 1, PSTATE.SS == 0: (active-pending)

     *   emit code to generate a software step exception

     *   end the TB

     */

    dc->ss_active = ARM_TBFLAG_SS_ACTIVE(dc->base.tb->flags);

    dc->pstate_ss = ARM_TBFLAG_PSTATE_SS(dc->base.tb->flags);

    dc->is_ldex = false;

    dc->ss_same_el = (arm_debug_target_el(env) == dc->current_el);



    dc->next_page_start =

        (dc->base.pc_first & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;



    init_tmp_a64_array(dc);



    return max_insns;

}
