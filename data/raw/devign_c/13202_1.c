static void gen_write_xer(TCGv src)

{

    tcg_gen_andi_tl(cpu_xer, src,

                    ~((1u << XER_SO) | (1u << XER_OV) | (1u << XER_CA)));

    tcg_gen_extract_tl(cpu_so, src, XER_SO, 1);

    tcg_gen_extract_tl(cpu_ov, src, XER_OV, 1);

    tcg_gen_extract_tl(cpu_ca, src, XER_CA, 1);

}
