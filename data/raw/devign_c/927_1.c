static void arm_tr_tb_stop(DisasContextBase *dcbase, CPUState *cpu)

{

    DisasContext *dc = container_of(dcbase, DisasContext, base);



    if (dc->base.tb->cflags & CF_LAST_IO && dc->condjmp) {

        /* FIXME: This can theoretically happen with self-modifying code. */

        cpu_abort(cpu, "IO on conditional branch instruction");

    }



    /* At this stage dc->condjmp will only be set when the skipped

       instruction was a conditional branch or trap, and the PC has

       already been written.  */

    gen_set_condexec(dc);

    if (dc->base.is_jmp == DISAS_BX_EXCRET) {

        /* Exception return branches need some special case code at the

         * end of the TB, which is complex enough that it has to

         * handle the single-step vs not and the condition-failed

         * insn codepath itself.

         */

        gen_bx_excret_final_code(dc);

    } else if (unlikely(is_singlestepping(dc))) {

        /* Unconditional and "condition passed" instruction codepath. */

        switch (dc->base.is_jmp) {

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

        case DISAS_TOO_MANY:

        case DISAS_UPDATE:

            gen_set_pc_im(dc, dc->pc);

            /* fall through */

        default:

            /* FIXME: Single stepping a WFI insn will not halt the CPU. */

            gen_singlestep_exception(dc);

            break;

        case DISAS_NORETURN:

            break;

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

        switch(dc->base.is_jmp) {

        case DISAS_NEXT:

        case DISAS_TOO_MANY:

            gen_goto_tb(dc, 1, dc->pc);

            break;

        case DISAS_JUMP:

            gen_goto_ptr();

            break;

        case DISAS_UPDATE:

            gen_set_pc_im(dc, dc->pc);

            /* fall through */

        default:

            /* indicate that the hash table must be used to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_NORETURN:

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



    /* Functions above can change dc->pc, so re-align db->pc_next */

    dc->base.pc_next = dc->pc;

}
