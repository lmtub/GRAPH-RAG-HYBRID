static void gen_adc(TCGv t0, TCGv t1)

{

    TCGv tmp;

    tcg_gen_add_i32(t0, t0, t1);

    tmp = load_cpu_field(CF);

    tcg_gen_add_i32(t0, t0, tmp);

    dead_tmp(tmp);

}
