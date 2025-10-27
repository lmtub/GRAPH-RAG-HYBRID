static void arm_tr_translate_insn(DisasContextBase *dcbase, CPUState *cpu)

{

    DisasContext *dc = container_of(dcbase, DisasContext, base);

    CPUARMState *env = cpu->env_ptr;



#ifdef CONFIG_USER_ONLY

    /* Intercept jump to the magic kernel page.  */

    if (dc->pc >= 0xffff0000) {

        /* We always get here via a jump, so know we are not in a

           conditional execution block.  */

        gen_exception_internal(EXCP_KERNEL_TRAP);

        dc->base.is_jmp = DISAS_NORETURN;

        return;

    }

#endif



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

        assert(dc->base.num_insns == 1);

        gen_exception(EXCP_UDEF, syn_swstep(dc->ss_same_el, 0, 0),

                      default_exception_el(dc));

        dc->base.is_jmp = DISAS_NORETURN;

        return;

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



    if (dc->condjmp && !dc->base.is_jmp) {

        gen_set_label(dc->condlabel);

        dc->condjmp = 0;

    }



    if (dc->base.is_jmp == DISAS_NEXT) {

        /* Translation stops when a conditional branch is encountered.

         * Otherwise the subsequent code could get translated several times.

         * Also stop translation when a page boundary is reached.  This

         * ensures prefetch aborts occur at the right place.  */



        if (dc->pc >= dc->next_page_start ||

            (dc->pc >= dc->next_page_start - 3 &&

             insn_crosses_page(env, dc))) {

            /* We want to stop the TB if the next insn starts in a new page,

             * or if it spans between this page and the next. This means that

             * if we're looking at the last halfword in the page we need to

             * see if it's a 16-bit Thumb insn (which will fit in this TB)

             * or a 32-bit Thumb insn (which won't).

             * This is to avoid generating a silly TB with a single 16-bit insn

             * in it at the end of this page (which would execute correctly

             * but isn't very efficient).

             */

            dc->base.is_jmp = DISAS_TOO_MANY;

        }

    }



    dc->base.pc_next = dc->pc;

    translator_loop_temp_check(&dc->base);

}
