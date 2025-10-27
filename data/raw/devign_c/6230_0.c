static inline void gen_neon_negl(TCGv var, int size)

{

    switch (size) {

    case 0: gen_helper_neon_negl_u16(var, var); break;

    case 1: gen_helper_neon_negl_u32(var, var); break;

    case 2: gen_helper_neon_negl_u64(var, var); break;

    default: abort();

    }

}
