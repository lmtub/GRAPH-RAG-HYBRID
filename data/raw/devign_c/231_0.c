static inline void sync_jmpstate(DisasContext *dc)

{

    if (dc->jmp == JMP_DIRECT) {

            dc->jmp = JMP_INDIRECT;

            tcg_gen_movi_tl(env_btaken, 1);

            tcg_gen_movi_tl(env_btarget, dc->jmp_pc);

    }

}
