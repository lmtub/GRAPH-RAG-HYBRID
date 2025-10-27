static void gen_thumb2_parallel_addsub(int op1, int op2, TCGv a, TCGv b)

{

    TCGv tmp;



    switch (op1) {

#define gen_pas_helper(name) glue(gen_helper_,name)(a, a, b, tmp)

    case 0:

        tmp = tcg_temp_new(TCG_TYPE_PTR);

        tcg_gen_addi_ptr(tmp, cpu_env, offsetof(CPUState, GE));

        PAS_OP(s)

        break;

    case 4:

        tmp = tcg_temp_new(TCG_TYPE_PTR);

        tcg_gen_addi_ptr(tmp, cpu_env, offsetof(CPUState, GE));

        PAS_OP(u)

        break;

#undef gen_pas_helper

#define gen_pas_helper(name) glue(gen_helper_,name)(a, a, b)

    case 1:

        PAS_OP(q);

        break;

    case 2:

        PAS_OP(sh);

        break;

    case 5:

        PAS_OP(uq);

        break;

    case 6:

        PAS_OP(uh);

        break;

#undef gen_pas_helper

    }

}
