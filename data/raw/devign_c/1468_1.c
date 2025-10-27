static void gen_sub_carry(TCGv dest, TCGv t0, TCGv t1)

{

    TCGv tmp;

    tcg_gen_sub_i32(dest, t0, t1);

    tmp = load_cpu_field(CF);

    tcg_gen_add_i32(dest, dest, tmp);

    tcg_gen_subi_i32(dest, dest, 1);

    dead_tmp(tmp);

}
