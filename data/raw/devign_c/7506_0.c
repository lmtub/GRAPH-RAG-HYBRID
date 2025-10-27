static void disas_ldst_excl(DisasContext *s, uint32_t insn)

{

    int rt = extract32(insn, 0, 5);

    int rn = extract32(insn, 5, 5);

    int rt2 = extract32(insn, 10, 5);

    int is_lasr = extract32(insn, 15, 1);

    int rs = extract32(insn, 16, 5);

    int is_pair = extract32(insn, 21, 1);

    int is_store = !extract32(insn, 22, 1);

    int is_excl = !extract32(insn, 23, 1);

    int size = extract32(insn, 30, 2);

    TCGv_i64 tcg_addr;



    if ((!is_excl && !is_lasr) ||

        (is_pair && size < 2)) {

        unallocated_encoding(s);

        return;

    }



    if (rn == 31) {

        gen_check_sp_alignment(s);

    }

    tcg_addr = read_cpu_reg_sp(s, rn, 1);



    /* Note that since TCG is single threaded load-acquire/store-release

     * semantics require no extra if (is_lasr) { ... } handling.

     */



    if (is_excl) {

        if (!is_store) {

            s->is_ldex = true;

            gen_load_exclusive(s, rt, rt2, tcg_addr, size, is_pair);

        } else {

            gen_store_exclusive(s, rs, rt, rt2, tcg_addr, size, is_pair);

        }

    } else {

        TCGv_i64 tcg_rt = cpu_reg(s, rt);

        if (is_store) {

            do_gpr_st(s, tcg_rt, tcg_addr, size);

        } else {

            do_gpr_ld(s, tcg_rt, tcg_addr, size, false, false);

        }

        if (is_pair) {

            TCGv_i64 tcg_rt2 = cpu_reg(s, rt);

            tcg_gen_addi_i64(tcg_addr, tcg_addr, 1 << size);

            if (is_store) {

                do_gpr_st(s, tcg_rt2, tcg_addr, size);

            } else {

                do_gpr_ld(s, tcg_rt2, tcg_addr, size, false, false);

            }

        }

    }

}
