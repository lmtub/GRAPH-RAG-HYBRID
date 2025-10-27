static inline void gen_neon_narrow_sats(int size, TCGv dest, TCGv src)

{

    switch (size) {

    case 0: gen_helper_neon_narrow_sat_s8(dest, cpu_env, src); break;

    case 1: gen_helper_neon_narrow_sat_s16(dest, cpu_env, src); break;

    case 2: gen_helper_neon_narrow_sat_s32(dest, cpu_env, src); break;

    default: abort();

    }

}
