static inline void gen_neon_addl_saturate(TCGv op0, TCGv op1, int size)

{

    switch (size) {

    case 1: gen_helper_neon_addl_saturate_s32(op0, cpu_env, op0, op1); break;

    case 2: gen_helper_neon_addl_saturate_s64(op0, cpu_env, op0, op1); break;

    default: abort();

    }

}
