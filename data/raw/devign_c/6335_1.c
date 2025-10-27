static inline void gen_st16(TCGv val, TCGv addr, int index)

{

    tcg_gen_qemu_st16(val, addr, index);

    dead_tmp(val);

}
