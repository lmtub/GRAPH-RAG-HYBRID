void gen_intermediate_code(CPUARMState *env, TranslationBlock *tb)

{

    ARMCPU *cpu = arm_env_get_cpu(env);

    CPUState *cs = CPU(cpu);

    DisasContext dc1, *dc = &dc1;

    target_ulong pc_start;

    target_ulong next_page_start;

    int num_insns;

    int max_insns;

    bool end_of_page;



    /* generate intermediate code */



    /* The A64 decoder has its own top level loop, because it doesn't need

     * the A32/T32 complexity to do with conditional execution/IT blocks/etc.

     */

    if (ARM_TBFLAG_AARCH64_STATE(tb->flags)) {

        gen_intermediate_code_a64(cpu, tb);

        return;

    }



    pc_start = tb->pc;



    dc->tb = tb;



    dc->is_jmp = DISAS_NEXT;

    dc->pc = pc_start;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->condjmp = 0;



    dc->aarch64 = 0;

    /* If we are coming from secure EL0 in a system with a 32-bit EL3, then

     * there is no secure EL1, so we route exceptions to EL3.

     */

    dc->secure_routed_to_el3 = arm_feature(env, ARM_FEATURE_EL3) &&

                               !arm_el_is_aa64(env, 3);

    dc->thumb = ARM_TBFLAG_THUMB(tb->flags);

    dc->sctlr_b = ARM_TBFLAG_SCTLR_B(tb->flags);

    dc->be_data = ARM_TBFLAG_BE_DATA(tb->flags) ? MO_BE : MO_LE;

    dc->condexec_mask = (ARM_TBFLAG_CONDEXEC(tb->flags) & 0xf) << 1;

    dc->condexec_cond = ARM_TBFLAG_CONDEXEC(tb->flags) >> 4;

    dc->mmu_idx = ARM_TBFLAG_MMUIDX(tb->flags);

    dc->current_el = arm_mmu_idx_to_el(dc->mmu_idx);

#if !defined(CONFIG_USER_ONLY)

    dc->user = (dc->current_el == 0);

#endif

    dc->ns = ARM_TBFLAG_NS(tb->flags);

    dc->fp_excp_el = ARM_TBFLAG_FPEXC_EL(tb->flags);

    dc->vfp_enabled = ARM_TBFLAG_VFPEN(tb->flags);

    dc->vec_len = ARM_TBFLAG_VECLEN(tb->flags);

    dc->vec_stride = ARM_TBFLAG_VECSTRIDE(tb->flags);

    dc->c15_cpar = ARM_TBFLAG_XSCALE_CPAR(tb->flags);


    dc->cp_regs = cpu->cp_regs;

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

    dc->ss_active = ARM_TBFLAG_SS_ACTIVE(tb->flags);

    dc->pstate_ss = ARM_TBFLAG_PSTATE_SS(tb->flags);

    dc->is_ldex = false;

    dc->ss_same_el = false; /* Can't be true since EL_d must be AArch64 */



    cpu_F0s = tcg_temp_new_i32();

    cpu_F1s = tcg_temp_new_i32();

    cpu_F0d = tcg_temp_new_i64();

    cpu_F1d = tcg_temp_new_i64();

    cpu_V0 = cpu_F0d;

    cpu_V1 = cpu_F1d;

    /* FIXME: cpu_M0 can probably be the same as cpu_V0.  */

    cpu_M0 = tcg_temp_new_i64();

    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }

    if (max_insns > TCG_MAX_INSNS) {

        max_insns = TCG_MAX_INSNS;

    }



    gen_tb_start(tb);



    tcg_clear_temp_count();



    /* A note on handling of the condexec (IT) bits:

     *

     * We want to avoid the overhead of having to write the updated condexec

     * bits back to the CPUARMState for every instruction in an IT block. So:

     * (1) if the condexec bits are not already zero then we write

     * zero back into the CPUARMState now. This avoids complications trying

     * to do it at the end of the block. (For example if we don't do this

     * it's hard to identify whether we can safely skip writing condexec

     * at the end of the TB, which we definitely want to do for the case

     * where a TB doesn't do anything with the IT state at all.)

     * (2) if we are going to leave the TB then we call gen_set_condexec()

     * which will write the correct value into CPUARMState if zero is wrong.

     * This is done both for leaving the TB at the end, and for leaving

     * it because of an exception we know will happen, which is done in

     * gen_exception_insn(). The latter is necessary because we need to

     * leave the TB with the PC/IT state just prior to execution of the

     * instruction which caused the exception.

     * (3) if we leave the TB unexpectedly (eg a data abort on a load)

     * then the CPUARMState will be wrong and we need to reset it.

     * This is handled in the same way as restoration of the

     * PC in these situations; we save the value of the condexec bits

     * for each PC via tcg_gen_insn_start(), and restore_state_to_opc()

     * then uses this to restore them after an exception.

     *

     * Note that there are no instructions which can read the condexec

     * bits, and none which can write non-static values to them, so

     * we don't need to care about whether CPUARMState is correct in the

     * middle of a TB.

     */



    /* Reset the conditional execution bits immediately. This avoids

       complications trying to do it at the end of the block.  */

    if (dc->condexec_mask || dc->condexec_cond)

      {

        TCGv_i32 tmp = tcg_temp_new_i32();

        tcg_gen_movi_i32(tmp, 0);

        store_cpu_field(tmp, condexec_bits);

      }

    do {

        dc->insn_start_idx = tcg_op_buf_count();

        tcg_gen_insn_start(dc->pc,

                           (dc->condexec_cond << 4) | (dc->condexec_mask >> 1),

                           0);

        num_insns++;



#ifdef CONFIG_USER_ONLY

        /* Intercept jump to the magic kernel page.  */

        if (dc->pc >= 0xffff0000) {

            /* We always get here via a jump, so know we are not in a

               conditional execution block.  */

            gen_exception_internal(EXCP_KERNEL_TRAP);

            dc->is_jmp = DISAS_EXC;

            break;

        }

#else

        if (arm_dc_feature(dc, ARM_FEATURE_M)) {

            /* Branches to the magic exception-return addresses should

             * already have been caught via the arm_v7m_unassigned_access hook,

             * and never get here.

             */

            assert(dc->pc < 0xfffffff0);

        }

#endif



        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            CPUBreakpoint *bp;

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (bp->pc == dc->pc) {

                    if (bp->flags & BP_CPU) {

                        gen_set_condexec(dc);

                        gen_set_pc_im(dc, dc->pc);

                        gen_helper_check_breakpoints(cpu_env);

                        /* End the TB early; it's likely not going to be executed */

                        dc->is_jmp = DISAS_UPDATE;

                    } else {

                        gen_exception_internal_insn(dc, 0, EXCP_DEBUG);

                        /* The address covered by the breakpoint must be

                           included in [tb->pc, tb->pc + tb->size) in order

                           to for it to be properly cleared -- thus we

                           increment the PC here so that the logic setting

                           tb->size below does the right thing.  */

                        /* TODO: Advance PC by correct instruction length to

                         * avoid disassembler error messages */

                        dc->pc += 2;

                        goto done_generating;

                    }

                    break;

                }

            }

        }



        if (num_insns == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }



        if (dc->ss_active && !dc->pstate_ss) {

            /* Singlestep state is Active-pending.

             * If we're in this state at the start of a TB then either

             *  a) we just took an exception to an EL which is being debugged

             *     and this is the first insn in the exception handler

             *  b) debug exceptions were masked and we just unmasked them

             *     without changing EL (eg by clearing PSTATE.D)

             * In either case we're going to take a swstep exception in the

             * "did not step an insn" case, and so the syndrome ISV and EX

             * bits should be zero.

             */

            assert(num_insns == 1);

            gen_exception(EXCP_UDEF, syn_swstep(dc->ss_same_el, 0, 0),

                          default_exception_el(dc));

            goto done_generating;

        }



        if (dc->thumb) {

            disas_thumb_insn(env, dc);

            if (dc->condexec_mask) {

                dc->condexec_cond = (dc->condexec_cond & 0xe)

                                   | ((dc->condexec_mask >> 4) & 1);

                dc->condexec_mask = (dc->condexec_mask << 1) & 0x1f;

                if (dc->condexec_mask == 0) {

                    dc->condexec_cond = 0;

                }

            }

        } else {

            unsigned int insn = arm_ldl_code(env, dc->pc, dc->sctlr_b);

            dc->pc += 4;

            disas_arm_insn(dc, insn);

        }



        if (dc->condjmp && !dc->is_jmp) {

            gen_set_label(dc->condlabel);

            dc->condjmp = 0;

        }



        if (tcg_check_temp_count()) {

            fprintf(stderr, "TCG temporary leak before "TARGET_FMT_lx"\n",

                    dc->pc);

        }



        /* Translation stops when a conditional branch is encountered.

         * Otherwise the subsequent code could get translated several times.

         * Also stop translation when a page boundary is reached.  This

         * ensures prefetch aborts occur at the right place.  */



        /* We want to stop the TB if the next insn starts in a new page,

         * or if it spans between this page and the next. This means that

         * if we're looking at the last halfword in the page we need to

         * see if it's a 16-bit Thumb insn (which will fit in this TB)

         * or a 32-bit Thumb insn (which won't).

         * This is to avoid generating a silly TB with a single 16-bit insn

         * in it at the end of this page (which would execute correctly

         * but isn't very efficient).

         */

        end_of_page = (dc->pc >= next_page_start) ||

            ((dc->pc >= next_page_start - 3) && insn_crosses_page(env, dc));



    } while (!dc->is_jmp && !tcg_op_buf_full() &&

             !is_singlestepping(dc) &&

             !singlestep &&

             !end_of_page &&

             num_insns < max_insns);



    if (tb->cflags & CF_LAST_IO) {

        if (dc->condjmp) {

            /* FIXME:  This can theoretically happen with self-modifying

               code.  */

            cpu_abort(cs, "IO on conditional branch instruction");

        }

        gen_io_end();

    }



    /* At this stage dc->condjmp will only be set when the skipped

       instruction was a conditional branch or trap, and the PC has

       already been written.  */

    gen_set_condexec(dc);

    if (unlikely(is_singlestepping(dc))) {

        /* Unconditional and "condition passed" instruction codepath. */

        switch (dc->is_jmp) {

        case DISAS_SWI:

            gen_ss_advance(dc);

            gen_exception(EXCP_SWI, syn_aa32_svc(dc->svc_imm, dc->thumb),

                          default_exception_el(dc));

            break;

        case DISAS_HVC:

            gen_ss_advance(dc);

            gen_exception(EXCP_HVC, syn_aa32_hvc(dc->svc_imm), 2);

            break;

        case DISAS_SMC:

            gen_ss_advance(dc);

            gen_exception(EXCP_SMC, syn_aa32_smc(), 3);

            break;

        case DISAS_NEXT:

        case DISAS_UPDATE:

            gen_set_pc_im(dc, dc->pc);

            /* fall through */

        default:

            /* FIXME: Single stepping a WFI insn will not halt the CPU. */

            gen_singlestep_exception(dc);

        }

    } else {

        /* While branches must always occur at the end of an IT block,

           there are a few other things that can cause us to terminate

           the TB in the middle of an IT block:

            - Exception generating instructions (bkpt, swi, undefined).

            - Page boundaries.

            - Hardware watchpoints.

           Hardware breakpoints have already been handled and skip this code.

         */

        switch(dc->is_jmp) {

        case DISAS_NEXT:

            gen_goto_tb(dc, 1, dc->pc);

            break;

        case DISAS_UPDATE:

            gen_set_pc_im(dc, dc->pc);

            /* fall through */

        case DISAS_JUMP:

        default:

            /* indicate that the hash table must be used to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_TB_JUMP:

            /* nothing more to generate */

            break;

        case DISAS_WFI:

            gen_helper_wfi(cpu_env);

            /* The helper doesn't necessarily throw an exception, but we

             * must go back to the main loop to check for interrupts anyway.

             */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_WFE:

            gen_helper_wfe(cpu_env);

            break;

        case DISAS_YIELD:

            gen_helper_yield(cpu_env);

            break;

        case DISAS_SWI:

            gen_exception(EXCP_SWI, syn_aa32_svc(dc->svc_imm, dc->thumb),

                          default_exception_el(dc));

            break;

        case DISAS_HVC:

            gen_exception(EXCP_HVC, syn_aa32_hvc(dc->svc_imm), 2);

            break;

        case DISAS_SMC:

            gen_exception(EXCP_SMC, syn_aa32_smc(), 3);

            break;

        }

    }



    if (dc->condjmp) {

        /* "Condition failed" instruction codepath for the branch/trap insn */

        gen_set_label(dc->condlabel);

        gen_set_condexec(dc);

        if (unlikely(is_singlestepping(dc))) {

            gen_set_pc_im(dc, dc->pc);

            gen_singlestep_exception(dc);

        } else {

            gen_goto_tb(dc, 1, dc->pc);

        }

    }



done_generating:

    gen_tb_end(tb, num_insns);



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM) &&

        qemu_log_in_addr_range(pc_start)) {

        qemu_log_lock();

        qemu_log("----------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(cs, pc_start, dc->pc - pc_start,

                         dc->thumb | (dc->sctlr_b << 1));

        qemu_log("\n");

        qemu_log_unlock();

    }

#endif

    tb->size = dc->pc - pc_start;

    tb->icount = num_insns;

}