static void arm_post_translate_insn(CPUARMState *env, DisasContext *dc)

{

    if (dc->condjmp && !dc->base.is_jmp) {

        gen_set_label(dc->condlabel);

        dc->condjmp = 0;

    }



    /* Translation stops when a conditional branch is encountered.

     * Otherwise the subsequent code could get translated several times.

     * Also stop translation when a page boundary is reached.  This

     * ensures prefetch aborts occur at the right place.

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



    dc->base.pc_next = dc->pc;

    translator_loop_temp_check(&dc->base);

}
