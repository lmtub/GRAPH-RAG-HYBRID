static void store_reg(DisasContext *s, int reg, TCGv var)

{

    if (reg == 15) {

        tcg_gen_andi_i32(var, var, ~1);

        s->is_jmp = DISAS_JUMP;

    }

    tcg_gen_mov_i32(cpu_R[reg], var);

    dead_tmp(var);

}
