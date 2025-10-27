static void tcg_out_brcond(TCGContext *s, TCGCond cond, TCGReg arg1,

                           TCGReg arg2, int label_index)

{

    static const MIPSInsn b_zero[16] = {

        [TCG_COND_LT] = OPC_BLTZ,

        [TCG_COND_GT] = OPC_BGTZ,

        [TCG_COND_LE] = OPC_BLEZ,

        [TCG_COND_GE] = OPC_BGEZ,

    };



    TCGLabel *l;

    MIPSInsn s_opc = OPC_SLTU;

    MIPSInsn b_opc;

    int cmp_map;



    switch (cond) {

    case TCG_COND_EQ:

        b_opc = OPC_BEQ;

        break;

    case TCG_COND_NE:

        b_opc = OPC_BNE;

        break;



    case TCG_COND_LT:

    case TCG_COND_GT:

    case TCG_COND_LE:

    case TCG_COND_GE:

        if (arg2 == 0) {

            b_opc = b_zero[cond];

            arg2 = arg1;

            arg1 = 0;

            break;

        }

        s_opc = OPC_SLT;

        /* FALLTHRU */



    case TCG_COND_LTU:

    case TCG_COND_GTU:

    case TCG_COND_LEU:

    case TCG_COND_GEU:

        cmp_map = mips_cmp_map[cond];

        if (cmp_map & MIPS_CMP_SWAP) {

            TCGReg t = arg1;

            arg1 = arg2;

            arg2 = t;

        }

        tcg_out_opc_reg(s, s_opc, TCG_TMP0, arg1, arg2);

        b_opc = (cmp_map & MIPS_CMP_INV ? OPC_BEQ : OPC_BNE);

        arg1 = TCG_TMP0;

        arg2 = TCG_REG_ZERO;

        break;



    default:

        tcg_abort();

        break;

    }



    tcg_out_opc_br(s, b_opc, arg1, arg2);

    l = &s->labels[label_index];

    if (l->has_value) {

        reloc_pc16(s->code_ptr - 1, l->u.value_ptr);

    } else {

        tcg_out_reloc(s, s->code_ptr - 1, R_MIPS_PC16, label_index, 0);

    }

    tcg_out_nop(s);

}
