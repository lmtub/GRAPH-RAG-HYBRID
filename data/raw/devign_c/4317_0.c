static void thumb_tr_translate_insn(DisasContextBase *dcbase, CPUState *cpu)

{

    DisasContext *dc = container_of(dcbase, DisasContext, base);

    CPUARMState *env = cpu->env_ptr;

    uint32_t insn;

    bool is_16bit;



    if (arm_pre_translate_insn(dc)) {

        return;

    }



    insn = arm_lduw_code(env, dc->pc, dc->sctlr_b);

    is_16bit = thumb_insn_is_16bit(dc, insn);

    dc->pc += 2;

    if (!is_16bit) {

        uint32_t insn2 = arm_lduw_code(env, dc->pc, dc->sctlr_b);



        insn = insn << 16 | insn2;

        dc->pc += 2;

    }

    dc->insn = insn;



    if (dc->condexec_mask && !thumb_insn_is_unconditional(dc, insn)) {

        uint32_t cond = dc->condexec_cond;



        if (cond != 0x0e) {     /* Skip conditional when condition is AL. */

            dc->condlabel = gen_new_label();

            arm_gen_test_cc(cond ^ 1, dc->condlabel);

            dc->condjmp = 1;

        }

    }



    if (is_16bit) {

        disas_thumb_insn(dc, insn);

    } else {

        if (disas_thumb2_insn(dc, insn)) {

            gen_exception_insn(dc, 4, EXCP_UDEF, syn_uncategorized(),

                               default_exception_el(dc));

        }

    }



    /* Advance the Thumb condexec condition.  */

    if (dc->condexec_mask) {

        dc->condexec_cond = ((dc->condexec_cond & 0xe) |

                             ((dc->condexec_mask >> 4) & 1));

        dc->condexec_mask = (dc->condexec_mask << 1) & 0x1f;

        if (dc->condexec_mask == 0) {

            dc->condexec_cond = 0;

        }

    }



    arm_post_translate_insn(dc);



    /* Thumb is a variable-length ISA.  Stop translation when the next insn

     * will touch a new page.  This ensures that prefetch aborts occur at

     * the right place.

     *

     * We want to stop the TB if the next insn starts in a new page,

     * or if it spans between this page and the next. This means that

     * if we're looking at the last halfword in the page we need to

     * see if it's a 16-bit Thumb insn (which will fit in this TB)

     * or a 32-bit Thumb insn (which won't).

     * This is to avoid generating a silly TB with a single 16-bit insn

     * in it at the end of this page (which would execute correctly

     * but isn't very efficient).

     */

    if (dc->base.is_jmp == DISAS_NEXT

        && (dc->pc >= dc->next_page_start

            || (dc->pc >= dc->next_page_start - 3

                && insn_crosses_page(env, dc)))) {

        dc->base.is_jmp = DISAS_TOO_MANY;

    }

}
