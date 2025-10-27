static inline void gen_st8(TCGv val, TCGv addr, int index)

{

    tcg_gen_qemu_st8(val, addr, index);

    dead_tmp(val);

}
