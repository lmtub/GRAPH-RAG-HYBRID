static void br(DisasContext *dc, uint32_t code, uint32_t flags)

{

    I_TYPE(instr, code);



    gen_goto_tb(dc, 0, dc->pc + 4 + (instr.imm16s & -4));

    dc->is_jmp = DISAS_TB_JUMP;

}
