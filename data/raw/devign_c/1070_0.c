void tcg_gen_mb(TCGBar mb_type)

{

    if (parallel_cpus) {

        tcg_gen_op1(INDEX_op_mb, mb_type);

    }

}
