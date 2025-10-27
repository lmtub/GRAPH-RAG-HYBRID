static void cris_alu_op_exec(DisasContext *dc, int op, 

                   TCGv dst, TCGv a, TCGv b, int size)

{

    /* Emit the ALU insns.  */

    switch (op) {

    case CC_OP_ADD:

        tcg_gen_add_tl(dst, a, b);

        /* Extended arithmetics.  */

        t_gen_addx_carry(dc, dst);

        break;

    case CC_OP_ADDC:

        tcg_gen_add_tl(dst, a, b);

        t_gen_add_flag(dst, 0); /* C_FLAG.  */

        break;

    case CC_OP_MCP:

        tcg_gen_add_tl(dst, a, b);

        t_gen_add_flag(dst, 8); /* R_FLAG.  */

        break;

    case CC_OP_SUB:

        tcg_gen_sub_tl(dst, a, b);

        /* Extended arithmetics.  */

        t_gen_subx_carry(dc, dst);

        break;

    case CC_OP_MOVE:

        tcg_gen_mov_tl(dst, b);

        break;

    case CC_OP_OR:

        tcg_gen_or_tl(dst, a, b);

        break;

    case CC_OP_AND:

        tcg_gen_and_tl(dst, a, b);

        break;

    case CC_OP_XOR:

        tcg_gen_xor_tl(dst, a, b);

        break;

    case CC_OP_LSL:

        t_gen_lsl(dst, a, b);

        break;

    case CC_OP_LSR:

        t_gen_lsr(dst, a, b);

        break;

    case CC_OP_ASR:

        t_gen_asr(dst, a, b);

        break;

    case CC_OP_NEG:

        tcg_gen_neg_tl(dst, b);

        /* Extended arithmetics.  */

        t_gen_subx_carry(dc, dst);

        break;

    case CC_OP_LZ:

        gen_helper_lz(dst, b);

        break;

    case CC_OP_MULS:

        tcg_gen_muls2_tl(dst, cpu_PR[PR_MOF], a, b);

        break;

    case CC_OP_MULU:

        tcg_gen_mulu2_tl(dst, cpu_PR[PR_MOF], a, b);

        break;

    case CC_OP_DSTEP:

        t_gen_cris_dstep(dst, a, b);

        break;

    case CC_OP_MSTEP:

        t_gen_cris_mstep(dst, a, b, cpu_PR[PR_CCS]);

        break;

    case CC_OP_BOUND:

    {

        int l1;

        l1 = gen_new_label();

        tcg_gen_mov_tl(dst, a);

        tcg_gen_brcond_tl(TCG_COND_LEU, a, b, l1);

        tcg_gen_mov_tl(dst, b);

        gen_set_label(l1);

    }

        break;

    case CC_OP_CMP:

        tcg_gen_sub_tl(dst, a, b);

        /* Extended arithmetics.  */

        t_gen_subx_carry(dc, dst);

        break;

    default:

        qemu_log("illegal ALU op.\n");

        BUG();

        break;

    }



    if (size == 1) {

        tcg_gen_andi_tl(dst, dst, 0xff);

    } else if (size == 2) {

        tcg_gen_andi_tl(dst, dst, 0xffff);

    }

}
