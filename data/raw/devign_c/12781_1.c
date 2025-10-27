gen_set_condexec (DisasContext *s)

{

    if (s->condexec_mask) {

        uint32_t val = (s->condexec_cond << 4) | (s->condexec_mask >> 1);

        TCGv tmp = new_tmp();

        tcg_gen_movi_i32(tmp, val);

        store_cpu_field(tmp, condexec_bits);

    }

}
