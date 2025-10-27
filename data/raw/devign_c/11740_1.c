static inline TCGv load_cpu_offset(int offset)

{

    TCGv tmp = new_tmp();

    tcg_gen_ld_i32(tmp, cpu_env, offset);

    return tmp;

}
