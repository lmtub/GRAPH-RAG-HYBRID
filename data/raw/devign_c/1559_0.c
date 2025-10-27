static void gen_farith (DisasContext *ctx, uint32_t op1,

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



    switch (ctx->opcode & FOP(0x3f, 0x1f)) {

    case FOP(0, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_add_s(fp0, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "add.s";

        optype = BINOP;

        break;

    case FOP(1, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_sub_s(fp0, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "sub.s";

        optype = BINOP;

        break;

    case FOP(2, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_mul_s(fp0, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "mul.s";

        optype = BINOP;

        break;

    case FOP(3, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_div_s(fp0, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "div.s";

        optype = BINOP;

        break;

    case FOP(4, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_sqrt_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "sqrt.s";

        break;

    case FOP(5, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_abs_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "abs.s";

        break;

    case FOP(6, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "mov.s";

        break;

    case FOP(7, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_chs_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "neg.s";

        break;

    case FOP(8, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_roundl_s(fp64, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "round.l.s";

        break;

    case FOP(9, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_truncl_s(fp64, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "trunc.l.s";

        break;

    case FOP(10, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_ceill_s(fp64, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "ceil.l.s";

        break;

    case FOP(11, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_floorl_s(fp64, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "floor.l.s";

        break;

    case FOP(12, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_roundw_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "round.w.s";

        break;

    case FOP(13, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_truncw_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "trunc.w.s";

        break;

    case FOP(14, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_ceilw_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "ceil.w.s";

        break;

    case FOP(15, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_floorw_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "floor.w.s";

        break;

    case FOP(17, 16):

        gen_movcf_s(fs, fd, (ft >> 2) & 0x7, ft & 0x1);

        opn = "movcf.s";

        break;

    case FOP(18, 16):

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

    case FOP(19, 16):

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

    case FOP(21, 16):

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_recip_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "recip.s";

        break;

    case FOP(22, 16):

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_rsqrt_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "rsqrt.s";

        break;

    case FOP(28, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, fd);

            gen_helper_float_recip2_s(fp0, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "recip2.s";

        break;

    case FOP(29, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_recip1_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "recip1.s";

        break;

    case FOP(30, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_rsqrt1_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "rsqrt1.s";

        break;

    case FOP(31, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_helper_float_rsqrt2_s(fp0, fp0, fp1);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "rsqrt2.s";

        break;

    case FOP(33, 16):

        check_cp1_registers(ctx, fd);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_cvtd_s(fp64, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.d.s";

        break;

    case FOP(36, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_cvtw_s(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.w.s";

        break;

    case FOP(37, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_cvtl_s(fp64, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.l.s";

        break;

    case FOP(38, 16):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp64 = tcg_temp_new_i64();

            TCGv_i32 fp32_0 = tcg_temp_new_i32();

            TCGv_i32 fp32_1 = tcg_temp_new_i32();



            gen_load_fpr32(fp32_0, fs);

            gen_load_fpr32(fp32_1, ft);

            tcg_gen_concat_i32_i64(fp64, fp32_0, fp32_1);

            tcg_temp_free_i32(fp32_1);

            tcg_temp_free_i32(fp32_0);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.ps.s";

        break;

    case FOP(48, 16):

    case FOP(49, 16):

    case FOP(50, 16):

    case FOP(51, 16):

    case FOP(52, 16):

    case FOP(53, 16):

    case FOP(54, 16):

    case FOP(55, 16):

    case FOP(56, 16):

    case FOP(57, 16):

    case FOP(58, 16):

    case FOP(59, 16):

    case FOP(60, 16):

    case FOP(61, 16):

    case FOP(62, 16):

    case FOP(63, 16):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            if (ctx->opcode & (1 << 6)) {

                check_cop1x(ctx);

                gen_cmpabs_s(func-48, fp0, fp1, cc);

                opn = condnames_abs[func-48];

            } else {

                gen_cmp_s(func-48, fp0, fp1, cc);

                opn = condnames[func-48];

            }

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

        }

        break;

    case FOP(0, 17):

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_add_d(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "add.d";

        optype = BINOP;

        break;

    case FOP(1, 17):

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_sub_d(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "sub.d";

        optype = BINOP;

        break;

    case FOP(2, 17):

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_mul_d(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mul.d";

        optype = BINOP;

        break;

    case FOP(3, 17):

        check_cp1_registers(ctx, fs | ft | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_div_d(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "div.d";

        optype = BINOP;

        break;

    case FOP(4, 17):

        check_cp1_registers(ctx, fs | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_sqrt_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "sqrt.d";

        break;

    case FOP(5, 17):

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

    case FOP(6, 17):

        check_cp1_registers(ctx, fs | fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mov.d";

        break;

    case FOP(7, 17):

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

    case FOP(8, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_roundl_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "round.l.d";

        break;

    case FOP(9, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_truncl_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "trunc.l.d";

        break;

    case FOP(10, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_ceill_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "ceil.l.d";

        break;

    case FOP(11, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_floorl_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "floor.l.d";

        break;

    case FOP(12, 17):

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_roundw_d(fp32, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "round.w.d";

        break;

    case FOP(13, 17):

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_truncw_d(fp32, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "trunc.w.d";

        break;

    case FOP(14, 17):

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_ceilw_d(fp32, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "ceil.w.d";

        break;

    case FOP(15, 17):

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_floorw_d(fp32, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "floor.w.d";

        break;

    case FOP(17, 17):

        gen_movcf_d(ctx, fs, fd, (ft >> 2) & 0x7, ft & 0x1);

        opn = "movcf.d";

        break;

    case FOP(18, 17):

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

    case FOP(19, 17):

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

    case FOP(21, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_recip_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip.d";

        break;

    case FOP(22, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_rsqrt_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt.d";

        break;

    case FOP(28, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_recip2_d(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip2.d";

        break;

    case FOP(29, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_recip1_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip1.d";

        break;

    case FOP(30, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_rsqrt1_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt1.d";

        break;

    case FOP(31, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_rsqrt2_d(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt2.d";

        break;

    case FOP(48, 17):

    case FOP(49, 17):

    case FOP(50, 17):

    case FOP(51, 17):

    case FOP(52, 17):

    case FOP(53, 17):

    case FOP(54, 17):

    case FOP(55, 17):

    case FOP(56, 17):

    case FOP(57, 17):

    case FOP(58, 17):

    case FOP(59, 17):

    case FOP(60, 17):

    case FOP(61, 17):

    case FOP(62, 17):

    case FOP(63, 17):

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            if (ctx->opcode & (1 << 6)) {

                check_cop1x(ctx);

                check_cp1_registers(ctx, fs | ft);

                gen_cmpabs_d(func-48, fp0, fp1, cc);

                opn = condnames_abs[func-48];

            } else {

                check_cp1_registers(ctx, fs | ft);

                gen_cmp_d(func-48, fp0, fp1, cc);

                opn = condnames[func-48];

            }

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

        }

        break;

    case FOP(32, 17):

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_cvts_d(fp32, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "cvt.s.d";

        break;

    case FOP(36, 17):

        check_cp1_registers(ctx, fs);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_cvtw_d(fp32, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "cvt.w.d";

        break;

    case FOP(37, 17):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtl_d(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.l.d";

        break;

    case FOP(32, 20):

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_cvts_w(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.s.w";

        break;

    case FOP(33, 20):

        check_cp1_registers(ctx, fd);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr32(fp32, fs);

            gen_helper_float_cvtd_w(fp64, fp32);

            tcg_temp_free_i32(fp32);

            gen_store_fpr64(ctx, fp64, fd);

            tcg_temp_free_i64(fp64);

        }

        opn = "cvt.d.w";

        break;

    case FOP(32, 21):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp32 = tcg_temp_new_i32();

            TCGv_i64 fp64 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp64, fs);

            gen_helper_float_cvts_l(fp32, fp64);

            tcg_temp_free_i64(fp64);

            gen_store_fpr32(fp32, fd);

            tcg_temp_free_i32(fp32);

        }

        opn = "cvt.s.l";

        break;

    case FOP(33, 21):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtd_l(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.d.l";

        break;

    case FOP(38, 20):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtps_pw(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.ps.pw";

        break;

    case FOP(0, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_add_ps(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "add.ps";

        break;

    case FOP(1, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_sub_ps(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "sub.ps";

        break;

    case FOP(2, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_mul_ps(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mul.ps";

        break;

    case FOP(5, 22):

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

    case FOP(6, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mov.ps";

        break;

    case FOP(7, 22):

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

    case FOP(17, 22):

        check_cp1_64bitmode(ctx);

        gen_movcf_ps(fs, fd, (ft >> 2) & 0x7, ft & 0x1);

        opn = "movcf.ps";

        break;

    case FOP(18, 22):

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

    case FOP(19, 22):

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

    case FOP(24, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, ft);

            gen_load_fpr64(ctx, fp1, fs);

            gen_helper_float_addr_ps(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "addr.ps";

        break;

    case FOP(26, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, ft);

            gen_load_fpr64(ctx, fp1, fs);

            gen_helper_float_mulr_ps(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "mulr.ps";

        break;

    case FOP(28, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, fd);

            gen_helper_float_recip2_ps(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip2.ps";

        break;

    case FOP(29, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_recip1_ps(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "recip1.ps";

        break;

    case FOP(30, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_rsqrt1_ps(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt1.ps";

        break;

    case FOP(31, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_helper_float_rsqrt2_ps(fp0, fp0, fp1);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "rsqrt2.ps";

        break;

    case FOP(32, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32h(fp0, fs);

            gen_helper_float_cvts_pu(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.s.pu";

        break;

    case FOP(36, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_helper_float_cvtpw_ps(fp0, fp0);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "cvt.pw.ps";

        break;

    case FOP(40, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_helper_float_cvts_pl(fp0, fp0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "cvt.s.pl";

        break;

    case FOP(44, 22):

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

    case FOP(45, 22):

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

    case FOP(46, 22):

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

    case FOP(47, 22):

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

    case FOP(48, 22):

    case FOP(49, 22):

    case FOP(50, 22):

    case FOP(51, 22):

    case FOP(52, 22):

    case FOP(53, 22):

    case FOP(54, 22):

    case FOP(55, 22):

    case FOP(56, 22):

    case FOP(57, 22):

    case FOP(58, 22):

    case FOP(59, 22):

    case FOP(60, 22):

    case FOP(61, 22):

    case FOP(62, 22):

    case FOP(63, 22):

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            if (ctx->opcode & (1 << 6)) {

                gen_cmpabs_ps(func-48, fp0, fp1, cc);

                opn = condnames_abs[func-48];

            } else {

                gen_cmp_ps(func-48, fp0, fp1, cc);

                opn = condnames[func-48];

            }

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

        }

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception (ctx, EXCP_RI);

        return;

    }

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
