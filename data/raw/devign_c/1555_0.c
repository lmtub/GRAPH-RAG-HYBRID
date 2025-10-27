static void gen_farith (DisasContext *ctx, enum fopcode op1,

                        int ft, int fs, int fd, int cc)

{

    const char *opn = "farith";

    const char *condnames[] = {

            "c.f",

            "c.un",

            "c.eq",

            "c.ueq",

            "c.olt",

            "c.ult",

            "c.ole",

            "c.ule",

            "c.sf",

            "c.ngle",

            "c.seq",

            "c.ngl",

            "c.lt",

            "c.nge",

            "c.le",

            "c.ngt",

    };

    const char *condnames_abs[] = {

            "cabs.f",

            "cabs.un",

            "cabs.eq",

            "cabs.ueq",

            "cabs.olt",

            "cabs.ult",

            "cabs.ole",

            "cabs.ule",

            "cabs.sf",

            "cabs.ngle",

            "cabs.seq",

            "cabs.ngl",

            "cabs.lt",

            "cabs.nge",

            "cabs.le",

            "cabs.ngt",

    };

    enum { BINOP, CMPOP, OTHEROP } optype = OTHEROP;

    uint32_t func = ctx->opcode & 0x3f;



    switch (op1) {

    case OPC_ADD_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_add_s(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "add.s";

        optype = BINOP;

        break;

    case OPC_SUB_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_sub_s(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "sub.s";

        optype = BINOP;

        break;

    case OPC_MUL_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_mul_s(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "mul.s";

        optype = BINOP;

        break;

    case OPC_DIV_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_div_s(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "div.s";

        optype = BINOP;

        break;

    case OPC_SQRT_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_sqrt_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "sqrt.s";

        break;

    case OPC_ABS_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_abs_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "abs.s";

        break;

    case OPC_MOV_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "mov.s";

        break;

    case OPC_NEG_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_chs_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "neg.s";

        break;

    case OPC_ROUND_L_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_roundl_s(fp64, cpu_env, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "round.l.s";

        break;

    case OPC_TRUNC_L_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_truncl_s(fp64, cpu_env, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "trunc.l.s";

        break;

    case OPC_CEIL_L_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_ceill_s(fp64, cpu_env, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "ceil.l.s";

        break;

    case OPC_FLOOR_L_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_floorl_s(fp64, cpu_env, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "floor.l.s";

        break;

    case OPC_ROUND_W_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_roundw_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "round.w.s";

        break;

    case OPC_TRUNC_W_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_truncw_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "trunc.w.s";

        break;

    case OPC_CEIL_W_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_ceilw_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "ceil.w.s";

        break;

    case OPC_FLOOR_W_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_floorw_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "floor.w.s";

        break;

    case OPC_MOVCF_S:

        gen_movcf_s(fs, fd, (ft >> 2) & 0x7, ft & 0x1);

        opn = "movcf.s";

        break;

    case OPC_MOVZ_S:

        {

            int l1 = gen_new_label();

            TCGv_i32 fp0;



            if (ft != 0) {

                tcg_gen_brcondi_tl(TCG_COND_NE, cpu_gpr[ft], 0, l1);

            }

            fp0 = tcg_temp_new_i32();

            gen_load_fpr32(fp0, fs);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

            gen_set_label(l1);

        }

        opn = "movz.s";

        break;

    case OPC_MOVN_S:

        {

            int l1 = gen_new_label();

            TCGv_i32 fp0;



            if (ft != 0) {

                tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_gpr[ft], 0, l1);

                fp0 = tcg_temp_new_i32();

                gen_load_fpr32(fp0, fs);

                gen_store_fpr32(fp0, fd);

                tcg_temp_free_i32(fp0);

                gen_set_label(l1);

            }

        }

        opn = "movn.s";

        break;

    case OPC_RECIP_S:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_recip_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "recip.s";

        break;

    case OPC_RSQRT_S:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_rsqrt_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "rsqrt.s";

        break;

    case OPC_RECIP2_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_recip2_s(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "recip2.s";

        break;

    case OPC_RECIP1_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_recip1_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "recip1.s";

        break;

    case OPC_RSQRT1_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_rsqrt1_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "rsqrt1.s";

        break;

    case OPC_RSQRT2_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_rsqrt2_s(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "rsqrt2.s";

        break;

    case OPC_CVT_D_S:

        check_cp1_registers(ctx, fd);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_cvtd_s(fp64, cpu_env, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.d.s";

        break;

    case OPC_CVT_W_S:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_cvtw_s(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.w.s";

        break;

    case OPC_CVT_L_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_cvtl_s(fp64, cpu_env, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.l.s";

        break;

    case OPC_CVT_PS_S:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp64 = tcg_temp_new_i64();

            TCGv_i32 fp32_0 = tcg_temp_new_i32();

            TCGv_i32 fp32_1 = tcg_temp_new_i32();



            gen_load_fpr32(fp32_0, fs);

            gen_load_fpr32(fp32_1, ft);

            tcg_gen_concat_i32_i64(fp64, fp32_1, fp32_0);

            tcg_temp_free_i32(fp32_1);

            tcg_temp_free_i32(fp32_0);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.ps.s";

        break;

    case OPC_CMP_F_S:

    case OPC_CMP_UN_S:

    case OPC_CMP_EQ_S:

    case OPC_CMP_UEQ_S:

    case OPC_CMP_OLT_S:

    case OPC_CMP_ULT_S:

    case OPC_CMP_OLE_S:

    case OPC_CMP_ULE_S:

    case OPC_CMP_SF_S:

    case OPC_CMP_NGLE_S:

    case OPC_CMP_SEQ_S:

    case OPC_CMP_NGL_S:

    case OPC_CMP_LT_S:

    case OPC_CMP_NGE_S:

    case OPC_CMP_LE_S:

    case OPC_CMP_NGT_S:

        if (ctx->opcode & (1 << 6)) {

            gen_cmpabs_s(ctx, func-48, ft, fs, cc);

            opn = condnames_abs[func-48];

        } else {

            gen_cmp_s(ctx, func-48, ft, fs, cc);

            opn = condnames[func-48];

        }

        break;

    case OPC_ADD_D:

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_add_d(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "add.d";

        optype = BINOP;

        break;

    case OPC_SUB_D:

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_sub_d(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "sub.d";

        optype = BINOP;

        break;

    case OPC_MUL_D:

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_mul_d(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mul.d";

        optype = BINOP;

        break;

    case OPC_DIV_D:

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_div_d(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "div.d";

        optype = BINOP;

        break;

    case OPC_SQRT_D:

        check_cp1_registers(ctx, fs | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_sqrt_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "sqrt.d";

        break;

    case OPC_ABS_D:

        check_cp1_registers(ctx, fs | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_abs_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "abs.d";

        break;

    case OPC_MOV_D:

        check_cp1_registers(ctx, fs | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mov.d";

        break;

    case OPC_NEG_D:

        check_cp1_registers(ctx, fs | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_chs_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "neg.d";

        break;

    case OPC_ROUND_L_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_roundl_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "round.l.d";

        break;

    case OPC_TRUNC_L_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_truncl_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "trunc.l.d";

        break;

    case OPC_CEIL_L_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_ceill_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "ceil.l.d";

        break;

    case OPC_FLOOR_L_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_floorl_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "floor.l.d";

        break;

    case OPC_ROUND_W_D:

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_roundw_d(fp32, cpu_env, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "round.w.d";

        break;

    case OPC_TRUNC_W_D:

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_truncw_d(fp32, cpu_env, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "trunc.w.d";

        break;

    case OPC_CEIL_W_D:

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_ceilw_d(fp32, cpu_env, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "ceil.w.d";

        break;

    case OPC_FLOOR_W_D:

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_floorw_d(fp32, cpu_env, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "floor.w.d";

        break;

    case OPC_MOVCF_D:

        gen_movcf_d(ctx, fs, fd, (ft >> 2) & 0x7, ft & 0x1);

        opn = "movcf.d";

        break;

    case OPC_MOVZ_D:

        {

            int l1 = gen_new_label();

            TCGv_i64 fp0;



            if (ft != 0) {

                tcg_gen_brcondi_tl(TCG_COND_NE, cpu_gpr[ft], 0, l1);

            }

            fp0 = tcg_temp_new_i64();

            gen_load_fpr64(ctx, fp0, fs);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

            gen_set_label(l1);

        }

        opn = "movz.d";

        break;

    case OPC_MOVN_D:

        {

            int l1 = gen_new_label();

            TCGv_i64 fp0;



            if (ft != 0) {

                tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_gpr[ft], 0, l1);

                fp0 = tcg_temp_new_i64();

                gen_load_fpr64(ctx, fp0, fs);

                gen_store_fpr64(ctx, fp0, fd);

                tcg_temp_free_i64(fp0);

                gen_set_label(l1);

            }

        }

        opn = "movn.d";

        break;

    case OPC_RECIP_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_recip_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip.d";

        break;

    case OPC_RSQRT_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_rsqrt_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt.d";

        break;

    case OPC_RECIP2_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_recip2_d(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip2.d";

        break;

    case OPC_RECIP1_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_recip1_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip1.d";

        break;

    case OPC_RSQRT1_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_rsqrt1_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt1.d";

        break;

    case OPC_RSQRT2_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_rsqrt2_d(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt2.d";

        break;

    case OPC_CMP_F_D:

    case OPC_CMP_UN_D:

    case OPC_CMP_EQ_D:

    case OPC_CMP_UEQ_D:

    case OPC_CMP_OLT_D:

    case OPC_CMP_ULT_D:

    case OPC_CMP_OLE_D:

    case OPC_CMP_ULE_D:

    case OPC_CMP_SF_D:

    case OPC_CMP_NGLE_D:

    case OPC_CMP_SEQ_D:

    case OPC_CMP_NGL_D:

    case OPC_CMP_LT_D:

    case OPC_CMP_NGE_D:

    case OPC_CMP_LE_D:

    case OPC_CMP_NGT_D:

        if (ctx->opcode & (1 << 6)) {

            gen_cmpabs_d(ctx, func-48, ft, fs, cc);

            opn = condnames_abs[func-48];

        } else {

            gen_cmp_d(ctx, func-48, ft, fs, cc);

            opn = condnames[func-48];

        }

        break;

    case OPC_CVT_S_D:

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_cvts_d(fp32, cpu_env, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "cvt.s.d";

        break;

    case OPC_CVT_W_D:

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_cvtw_d(fp32, cpu_env, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "cvt.w.d";

        break;

    case OPC_CVT_L_D:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtl_d(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.l.d";

        break;

    case OPC_CVT_S_W:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_cvts_w(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.s.w";

        break;

    case OPC_CVT_D_W:

        check_cp1_registers(ctx, fd);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_cvtd_w(fp64, cpu_env, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.d.w";

        break;

    case OPC_CVT_S_L:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_cvts_l(fp32, cpu_env, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "cvt.s.l";

        break;

    case OPC_CVT_D_L:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtd_l(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.d.l";

        break;

    case OPC_CVT_PS_PW:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtps_pw(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.ps.pw";

        break;

    case OPC_ADD_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_add_ps(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "add.ps";

        break;

    case OPC_SUB_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_sub_ps(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "sub.ps";

        break;

    case OPC_MUL_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_mul_ps(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mul.ps";

        break;

    case OPC_ABS_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_abs_ps(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "abs.ps";

        break;

    case OPC_MOV_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mov.ps";

        break;

    case OPC_NEG_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_chs_ps(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "neg.ps";

        break;

    case OPC_MOVCF_PS:

        check_cp1_64bitmode(ctx);

        gen_movcf_ps(fs, fd, (ft >> 2) & 0x7, ft & 0x1);

        opn = "movcf.ps";

        break;

    case OPC_MOVZ_PS:

        check_cp1_64bitmode(ctx);

        {

            int l1 = gen_new_label();

            TCGv_i64 fp0;



            if (ft != 0)

                tcg_gen_brcondi_tl(TCG_COND_NE, cpu_gpr[ft], 0, l1);

            fp0 = tcg_temp_new_i64();

            gen_load_fpr64(ctx, fp0, fs);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

            gen_set_label(l1);

        }

        opn = "movz.ps";

        break;

    case OPC_MOVN_PS:

        check_cp1_64bitmode(ctx);

        {

            int l1 = gen_new_label();

            TCGv_i64 fp0;



            if (ft != 0) {

                tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_gpr[ft], 0, l1);

                fp0 = tcg_temp_new_i64();

                gen_load_fpr64(ctx, fp0, fs);

                gen_store_fpr64(ctx, fp0, fd);

                tcg_temp_free_i64(fp0);

                gen_set_label(l1);

            }

        }

        opn = "movn.ps";

        break;

    case OPC_ADDR_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, ft);

            gen_load_fpr64(ctx, fp1, fs);

            gen_helper_float_addr_ps(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "addr.ps";

        break;

    case OPC_MULR_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, ft);

            gen_load_fpr64(ctx, fp1, fs);

            gen_helper_float_mulr_ps(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mulr.ps";

        break;

    case OPC_RECIP2_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_recip2_ps(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip2.ps";

        break;

    case OPC_RECIP1_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_recip1_ps(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip1.ps";

        break;

    case OPC_RSQRT1_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_rsqrt1_ps(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt1.ps";

        break;

    case OPC_RSQRT2_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_rsqrt2_ps(fp0, cpu_env, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt2.ps";

        break;

    case OPC_CVT_S_PU:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32h(fp0, fs);

            gen_helper_float_cvts_pu(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.s.pu";

        break;

    case OPC_CVT_PW_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtpw_ps(fp0, cpu_env, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.pw.ps";

        break;

    case OPC_CVT_S_PL:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_cvts_pl(fp0, cpu_env, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.s.pl";

        break;

    case OPC_PLL_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_store_fpr32h(fp0, fd);

            gen_store_fpr32(fp1, fd);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

        }

        opn = "pll.ps";

        break;

    case OPC_PLU_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32h(fp1, ft);

            gen_store_fpr32(fp1, fd);

            gen_store_fpr32h(fp0, fd);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

        }

        opn = "plu.ps";

        break;

    case OPC_PUL_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32h(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_store_fpr32(fp1, fd);

            gen_store_fpr32h(fp0, fd);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

        }

        opn = "pul.ps";

        break;

    case OPC_PUU_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32h(fp0, fs);

            gen_load_fpr32h(fp1, ft);

            gen_store_fpr32(fp1, fd);

            gen_store_fpr32h(fp0, fd);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

        }

        opn = "puu.ps";

        break;

    case OPC_CMP_F_PS:

    case OPC_CMP_UN_PS:

    case OPC_CMP_EQ_PS:

    case OPC_CMP_UEQ_PS:

    case OPC_CMP_OLT_PS:

    case OPC_CMP_ULT_PS:

    case OPC_CMP_OLE_PS:

    case OPC_CMP_ULE_PS:

    case OPC_CMP_SF_PS:

    case OPC_CMP_NGLE_PS:

    case OPC_CMP_SEQ_PS:

    case OPC_CMP_NGL_PS:

    case OPC_CMP_LT_PS:

    case OPC_CMP_NGE_PS:

    case OPC_CMP_LE_PS:

    case OPC_CMP_NGT_PS:

        if (ctx->opcode & (1 << 6)) {

            gen_cmpabs_ps(ctx, func-48, ft, fs, cc);

            opn = condnames_abs[func-48];

        } else {

            gen_cmp_ps(ctx, func-48, ft, fs, cc);

            opn = condnames[func-48];

        }

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception (ctx, EXCP_RI);

        return;

    }

    (void)opn; /* avoid a compiler warning */

    switch (optype) {

    case BINOP:

        MIPS_DEBUG("%s %s, %s, %s", opn, fregnames[fd], fregnames[fs], fregnames[ft]);

        break;

    case CMPOP:

        MIPS_DEBUG("%s %s,%s", opn, fregnames[fs], fregnames[ft]);

        break;

    default:

        MIPS_DEBUG("%s %s,%s", opn, fregnames[fd], fregnames[fs]);

        break;

    }

}
