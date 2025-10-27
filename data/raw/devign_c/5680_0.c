static inline TCGv *compute_ldst_addr(DisasContext *dc, TCGv *t)

{

    unsigned int extimm = dc->tb_flags & IMM_FLAG;

    /* Should be set to one if r1 is used by loadstores.  */

    int stackprot = 0;



    /* All load/stores use ra.  */

    if (dc->ra == 1) {

        stackprot = 1;

    }



    /* Treat the common cases first.  */

    if (!dc->type_b) {

        /* If any of the regs is r0, return a ptr to the other.  */

        if (dc->ra == 0) {

            return &cpu_R[dc->rb];

        } else if (dc->rb == 0) {

            return &cpu_R[dc->ra];

        }



        if (dc->rb == 1) {

            stackprot = 1;

        }



        *t = tcg_temp_new();

        tcg_gen_add_tl(*t, cpu_R[dc->ra], cpu_R[dc->rb]);



        if (stackprot) {

            gen_helper_stackprot(cpu_env, *t);

        }

        return t;

    }

    /* Immediate.  */

    if (!extimm) {

        if (dc->imm == 0) {

            return &cpu_R[dc->ra];

        }

        *t = tcg_temp_new();

        tcg_gen_movi_tl(*t, (int32_t)((int16_t)dc->imm));

        tcg_gen_add_tl(*t, cpu_R[dc->ra], *t);

    } else {

        *t = tcg_temp_new();

        tcg_gen_add_tl(*t, cpu_R[dc->ra], *(dec_alu_op_b(dc)));

    }



    if (stackprot) {

        gen_helper_stackprot(cpu_env, *t);

    }

    return t;

}
