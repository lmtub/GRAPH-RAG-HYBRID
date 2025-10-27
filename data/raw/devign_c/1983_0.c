static inline void gen_neon_widen(TCGv dest, TCGv src, int size, int u)

{

    if (u) {

        switch (size) {

        case 0: gen_helper_neon_widen_u8(dest, src); break;

        case 1: gen_helper_neon_widen_u16(dest, src); break;

        case 2: tcg_gen_extu_i32_i64(dest, src); break;

        default: abort();

        }

    } else {

        switch (size) {

        case 0: gen_helper_neon_widen_s8(dest, src); break;

        case 1: gen_helper_neon_widen_s16(dest, src); break;

        case 2: tcg_gen_ext_i32_i64(dest, src); break;

        default: abort();

        }

    }

    dead_tmp(src);

}
