static inline TCGv gen_ld16u(TCGv addr, int index)

{

    TCGv tmp = new_tmp();

    tcg_gen_qemu_ld16u(tmp, addr, index);

    return tmp;

}
