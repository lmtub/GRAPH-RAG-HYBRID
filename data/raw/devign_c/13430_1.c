static inline TCGv gen_ld16s(TCGv addr, int index)

{

    TCGv tmp = new_tmp();

    tcg_gen_qemu_ld16s(tmp, addr, index);

    return tmp;

}
