void disas_a64_insn(CPUARMState *env, DisasContext *s)

{

    uint32_t insn;



    insn = arm_ldl_code(env, s->pc, s->bswap_code);

    s->insn = insn;

    s->pc += 4;



    switch ((insn >> 24) & 0x1f) {

    default:

        unallocated_encoding(s);

        break;

    }



    if (unlikely(s->singlestep_enabled) && (s->is_jmp == DISAS_TB_JUMP)) {

        /* go through the main loop for single step */

        s->is_jmp = DISAS_JUMP;

    }

}
