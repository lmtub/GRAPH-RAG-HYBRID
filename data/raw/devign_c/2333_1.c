static inline void store_cpu_offset(TCGv var, int offset)

{

    tcg_gen_st_i32(var, cpu_env, offset);

    dead_tmp(var);

}
