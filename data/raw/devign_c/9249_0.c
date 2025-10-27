static inline void gen_neon_narrow(int size, TCGv dest, TCGv src)

{

    switch (size) {

    case 0: gen_helper_neon_narrow_u8(dest, src); break;

    case 1: gen_helper_neon_narrow_u16(dest, src); break;

    case 2: tcg_gen_trunc_i64_i32(dest, src); break;

    default: abort();

    }

}
