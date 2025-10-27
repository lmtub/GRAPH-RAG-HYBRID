static void gen_flt3_arith (DisasContext *ctx, uint32_t opc, int fd,

                            int fr, int fs, int ft)

{

    const char *opn = "flt3_arith";



    /* All of those work only on 64bit FPUs. */

    gen_op_cp1_64bitmode();

    switch (opc) {

    case OPC_ALNV_PS:

        GEN_LOAD_REG_TN(T0, fr);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        gen_op_float_alnv_ps();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "alnv.ps";

        break;

    case OPC_MADD_S:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        gen_op_float_muladd_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "madd.s";

        break;

    case OPC_MADD_D:

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        GEN_LOAD_FREG_FTN(DT2, fr);

        gen_op_float_muladd_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "madd.d";

        break;

    case OPC_MADD_PS:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        GEN_LOAD_FREG_FTN(WTH2, fr);

        gen_op_float_muladd_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "madd.ps";

        break;

    case OPC_MSUB_S:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        gen_op_float_mulsub_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "msub.s";

        break;

    case OPC_MSUB_D:

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        GEN_LOAD_FREG_FTN(DT2, fr);

        gen_op_float_mulsub_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "msub.d";

        break;

    case OPC_MSUB_PS:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        GEN_LOAD_FREG_FTN(WTH2, fr);

        gen_op_float_mulsub_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "msub.ps";

        break;

    case OPC_NMADD_S:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        gen_op_float_nmuladd_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "nmadd.s";

        break;

    case OPC_NMADD_D:

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        GEN_LOAD_FREG_FTN(DT2, fr);

        gen_op_float_nmuladd_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "nmadd.d";

        break;

    case OPC_NMADD_PS:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        GEN_LOAD_FREG_FTN(WTH2, fr);

        gen_op_float_nmuladd_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "nmadd.ps";

        break;

    case OPC_NMSUB_S:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        gen_op_float_nmulsub_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "nmsub.s";

        break;

    case OPC_NMSUB_D:

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        GEN_LOAD_FREG_FTN(DT2, fr);

        gen_op_float_nmulsub_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "nmsub.d";

        break;

    case OPC_NMSUB_PS:

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        GEN_LOAD_FREG_FTN(WT2, fr);

        GEN_LOAD_FREG_FTN(WTH2, fr);

        gen_op_float_nmulsub_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "nmsub.ps";

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception (ctx, EXCP_RI);

        return;

    }

    MIPS_DEBUG("%s %s, %s, %s, %s", opn, fregnames[fd], fregnames[fr],

               fregnames[fs], fregnames[ft]);

}
