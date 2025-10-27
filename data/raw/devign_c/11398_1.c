static inline void gen_st32(TCGv val, TCGv addr, int index)

{

    tcg_gen_qemu_st32(val, addr, index);

    dead_tmp(val);

}
