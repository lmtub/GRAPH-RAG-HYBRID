static void disas_uncond_b_reg(DisasContext *s, uint32_t insn)

{

    unsigned int opc, op2, op3, rn, op4;



    opc = extract32(insn, 21, 4);

    op2 = extract32(insn, 16, 5);

    op3 = extract32(insn, 10, 6);

    rn = extract32(insn, 5, 5);

    op4 = extract32(insn, 0, 5);



    if (op4 != 0x0 || op3 != 0x0 || op2 != 0x1f) {

        unallocated_encoding(s);


    }



    switch (opc) {

    case 0: /* BR */

    case 2: /* RET */

        break;

    case 1: /* BLR */

        tcg_gen_movi_i64(cpu_reg(s, 30), s->pc);

        break;

    case 4: /* ERET */




    case 5: /* DRPS */

        if (rn != 0x1f) {

            unallocated_encoding(s);

        } else {

            unsupported_encoding(s, insn);

        }


    default:

        unallocated_encoding(s);


    }



    tcg_gen_mov_i64(cpu_pc, cpu_reg(s, rn));


}