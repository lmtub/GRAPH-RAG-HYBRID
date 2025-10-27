void _decode_opc(DisasContext * ctx)

{

#if 0

    fprintf(stderr, "Translating opcode 0x%04x\n", ctx->opcode);

#endif

    switch (ctx->opcode) {

    case 0x0019:		/* div0u */

	tcg_gen_andi_i32(cpu_sr, cpu_sr, ~(SR_M | SR_Q | SR_T));

	return;

    case 0x000b:		/* rts */

	CHECK_NOT_DELAY_SLOT

	tcg_gen_mov_i32(cpu_delayed_pc, cpu_pr);

	ctx->flags |= DELAY_SLOT;

	ctx->delayed_pc = (uint32_t) - 1;

	return;

    case 0x0028:		/* clrmac */

	tcg_gen_movi_i32(cpu_mach, 0);

	tcg_gen_movi_i32(cpu_macl, 0);

	return;

    case 0x0048:		/* clrs */

	tcg_gen_andi_i32(cpu_sr, cpu_sr, ~SR_S);

	return;

    case 0x0008:		/* clrt */

	gen_clr_t();

	return;

    case 0x0038:		/* ldtlb */

#if defined(CONFIG_USER_ONLY)

	assert(0);		/* XXXXX */

#else

	tcg_gen_helper_0_0(helper_ldtlb);

#endif

	return;

    case 0x002b:		/* rte */

	CHECK_NOT_DELAY_SLOT

	tcg_gen_mov_i32(cpu_sr, cpu_ssr);

	tcg_gen_mov_i32(cpu_delayed_pc, cpu_spc);

	ctx->flags |= DELAY_SLOT;

	ctx->delayed_pc = (uint32_t) - 1;

	return;

    case 0x0058:		/* sets */

	tcg_gen_ori_i32(cpu_sr, cpu_sr, SR_S);

	return;

    case 0x0018:		/* sett */

	gen_set_t();

	return;

    case 0xfbfd:		/* frchg */

	tcg_gen_xori_i32(cpu_fpscr, cpu_fpscr, FPSCR_FR);

	ctx->bstate = BS_STOP;

	return;

    case 0xf3fd:		/* fschg */

	tcg_gen_xori_i32(cpu_fpscr, cpu_fpscr, FPSCR_SZ);

	ctx->bstate = BS_STOP;

	return;

    case 0x0009:		/* nop */

	return;

    case 0x001b:		/* sleep */

	if (ctx->memidx) {

		tcg_gen_helper_0_1(helper_sleep, tcg_const_i32(ctx->pc + 2));

	} else {

		tcg_gen_helper_0_0(helper_raise_illegal_instruction);

		ctx->bstate = BS_EXCP;

	}

	return;

    }



    switch (ctx->opcode & 0xf000) {

    case 0x1000:		/* mov.l Rm,@(disp,Rn) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, REG(B11_8), B3_0 * 4);

	    tcg_gen_qemu_st32(REG(B7_4), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x5000:		/* mov.l @(disp,Rm),Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, REG(B7_4), B3_0 * 4);

	    tcg_gen_qemu_ld32s(REG(B11_8), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xe000:		/* mov #imm,Rn */

	tcg_gen_movi_i32(REG(B11_8), B7_0s);

	return;

    case 0x9000:		/* mov.w @(disp,PC),Rn */

	{

	    TCGv addr = tcg_const_i32(ctx->pc + 4 + B7_0 * 2);

	    tcg_gen_qemu_ld16s(REG(B11_8), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xd000:		/* mov.l @(disp,PC),Rn */

	{

	    TCGv addr = tcg_const_i32((ctx->pc + 4 + B7_0 * 4) & ~3);

	    tcg_gen_qemu_ld32s(REG(B11_8), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x7000:		/* add #imm,Rn */

	tcg_gen_addi_i32(REG(B11_8), REG(B11_8), B7_0s);

	return;

    case 0xa000:		/* bra disp */

	CHECK_NOT_DELAY_SLOT

	ctx->delayed_pc = ctx->pc + 4 + B11_0s * 2;

	tcg_gen_movi_i32(cpu_delayed_pc, ctx->delayed_pc);

	ctx->flags |= DELAY_SLOT;

	return;

    case 0xb000:		/* bsr disp */

	CHECK_NOT_DELAY_SLOT

	tcg_gen_movi_i32(cpu_pr, ctx->pc + 4);

	ctx->delayed_pc = ctx->pc + 4 + B11_0s * 2;

	tcg_gen_movi_i32(cpu_delayed_pc, ctx->delayed_pc);

	ctx->flags |= DELAY_SLOT;

	return;

    }



    switch (ctx->opcode & 0xf00f) {

    case 0x6003:		/* mov Rm,Rn */

	tcg_gen_mov_i32(REG(B11_8), REG(B7_4));

	return;

    case 0x2000:		/* mov.b Rm,@Rn */

	tcg_gen_qemu_st8(REG(B7_4), REG(B11_8), ctx->memidx);

	return;

    case 0x2001:		/* mov.w Rm,@Rn */

	tcg_gen_qemu_st16(REG(B7_4), REG(B11_8), ctx->memidx);

	return;

    case 0x2002:		/* mov.l Rm,@Rn */

	tcg_gen_qemu_st32(REG(B7_4), REG(B11_8), ctx->memidx);

	return;

    case 0x6000:		/* mov.b @Rm,Rn */

	tcg_gen_qemu_ld8s(REG(B11_8), REG(B7_4), ctx->memidx);

	return;

    case 0x6001:		/* mov.w @Rm,Rn */

	tcg_gen_qemu_ld16s(REG(B11_8), REG(B7_4), ctx->memidx);

	return;

    case 0x6002:		/* mov.l @Rm,Rn */

	tcg_gen_qemu_ld32s(REG(B11_8), REG(B7_4), ctx->memidx);

	return;

    case 0x2004:		/* mov.b Rm,@-Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 1);

	    tcg_gen_qemu_st8(REG(B7_4), addr, ctx->memidx);	/* might cause re-execution */

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 1);	/* modify register status */

	    tcg_temp_free(addr);

	}

	return;

    case 0x2005:		/* mov.w Rm,@-Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 2);

	    tcg_gen_qemu_st16(REG(B7_4), addr, ctx->memidx);

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 2);

	    tcg_temp_free(addr);

	}

	return;

    case 0x2006:		/* mov.l Rm,@-Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 4);

	    tcg_gen_qemu_st32(REG(B7_4), addr, ctx->memidx);

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 4);

	}

	return;

    case 0x6004:		/* mov.b @Rm+,Rn */

	tcg_gen_qemu_ld8s(REG(B11_8), REG(B7_4), ctx->memidx);

	if ( B11_8 != B7_4 )

		tcg_gen_addi_i32(REG(B7_4), REG(B7_4), 1);

	return;

    case 0x6005:		/* mov.w @Rm+,Rn */

	tcg_gen_qemu_ld16s(REG(B11_8), REG(B7_4), ctx->memidx);

	if ( B11_8 != B7_4 )

		tcg_gen_addi_i32(REG(B7_4), REG(B7_4), 2);

	return;

    case 0x6006:		/* mov.l @Rm+,Rn */

	tcg_gen_qemu_ld32s(REG(B11_8), REG(B7_4), ctx->memidx);

	if ( B11_8 != B7_4 )

		tcg_gen_addi_i32(REG(B7_4), REG(B7_4), 4);

	return;

    case 0x0004:		/* mov.b Rm,@(R0,Rn) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B11_8), REG(0));

	    tcg_gen_qemu_st8(REG(B7_4), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x0005:		/* mov.w Rm,@(R0,Rn) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B11_8), REG(0));

	    tcg_gen_qemu_st16(REG(B7_4), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x0006:		/* mov.l Rm,@(R0,Rn) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B11_8), REG(0));

	    tcg_gen_qemu_st32(REG(B7_4), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x000c:		/* mov.b @(R0,Rm),Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B7_4), REG(0));

	    tcg_gen_qemu_ld8s(REG(B11_8), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x000d:		/* mov.w @(R0,Rm),Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B7_4), REG(0));

	    tcg_gen_qemu_ld16s(REG(B11_8), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x000e:		/* mov.l @(R0,Rm),Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B7_4), REG(0));

	    tcg_gen_qemu_ld32s(REG(B11_8), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x6008:		/* swap.b Rm,Rn */

	{

	    TCGv high, low;

	    high = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_ext8u_i32(high, REG(B7_4));

	    tcg_gen_shli_i32(high, high, 8);

	    low = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_shri_i32(low, REG(B7_4), 8);

	    tcg_gen_ext8u_i32(low, low);

	    tcg_gen_or_i32(REG(B11_8), high, low);

	    tcg_temp_free(low);

	    tcg_temp_free(high);

	}

	return;

    case 0x6009:		/* swap.w Rm,Rn */

	{

	    TCGv high, low;

	    high = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_ext16u_i32(high, REG(B7_4));

	    tcg_gen_shli_i32(high, high, 16);

	    low = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_shri_i32(low, REG(B7_4), 16);

	    tcg_gen_ext16u_i32(low, low);

	    tcg_gen_or_i32(REG(B11_8), high, low);

	    tcg_temp_free(low);

	    tcg_temp_free(high);

	}

	return;

    case 0x200d:		/* xtrct Rm,Rn */

	{

	    TCGv high, low;

	    high = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_ext16u_i32(high, REG(B7_4));

	    tcg_gen_shli_i32(high, high, 16);

	    low = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_shri_i32(low, REG(B11_8), 16);

	    tcg_gen_ext16u_i32(low, low);

	    tcg_gen_or_i32(REG(B11_8), high, low);

	    tcg_temp_free(low);

	    tcg_temp_free(high);

	}

	return;

    case 0x300c:		/* add Rm,Rn */

	tcg_gen_add_i32(REG(B11_8), REG(B11_8), REG(B7_4));

	return;

    case 0x300e:		/* addc Rm,Rn */

	tcg_gen_helper_1_2(helper_addc, REG(B11_8), REG(B7_4), REG(B11_8));

	return;

    case 0x300f:		/* addv Rm,Rn */

	tcg_gen_helper_1_2(helper_addv, REG(B11_8), REG(B7_4), REG(B11_8));

	return;

    case 0x2009:		/* and Rm,Rn */

	tcg_gen_and_i32(REG(B11_8), REG(B11_8), REG(B7_4));

	return;

    case 0x3000:		/* cmp/eq Rm,Rn */

	gen_cmp(TCG_COND_EQ, REG(B7_4), REG(B11_8));

	return;

    case 0x3003:		/* cmp/ge Rm,Rn */

	gen_cmp(TCG_COND_GE, REG(B7_4), REG(B11_8));

	return;

    case 0x3007:		/* cmp/gt Rm,Rn */

	gen_cmp(TCG_COND_GT, REG(B7_4), REG(B11_8));

	return;

    case 0x3006:		/* cmp/hi Rm,Rn */

	gen_cmp(TCG_COND_GTU, REG(B7_4), REG(B11_8));

	return;

    case 0x3002:		/* cmp/hs Rm,Rn */

	gen_cmp(TCG_COND_GEU, REG(B7_4), REG(B11_8));

	return;

    case 0x200c:		/* cmp/str Rm,Rn */

	{

	    int label1 = gen_new_label();

	    int label2 = gen_new_label();

	    TCGv cmp1 = tcg_temp_local_new(TCG_TYPE_I32);

	    TCGv cmp2 = tcg_temp_local_new(TCG_TYPE_I32);

	    tcg_gen_xor_i32(cmp1, REG(B7_4), REG(B11_8));

	    tcg_gen_andi_i32(cmp2, cmp1, 0xff000000);

	    tcg_gen_brcondi_i32(TCG_COND_EQ, cmp2, 0, label1);

	    tcg_gen_andi_i32(cmp2, cmp1, 0x00ff0000);

	    tcg_gen_brcondi_i32(TCG_COND_EQ, cmp2, 0, label1);

	    tcg_gen_andi_i32(cmp2, cmp1, 0x0000ff00);

	    tcg_gen_brcondi_i32(TCG_COND_EQ, cmp2, 0, label1);

	    tcg_gen_andi_i32(cmp2, cmp1, 0x000000ff);

	    tcg_gen_brcondi_i32(TCG_COND_EQ, cmp2, 0, label1);

	    tcg_gen_andi_i32(cpu_sr, cpu_sr, ~SR_T);

	    tcg_gen_br(label2);

	    gen_set_label(label1);

	    tcg_gen_ori_i32(cpu_sr, cpu_sr, SR_T);

	    gen_set_label(label2);

	    tcg_temp_free(cmp2);

	    tcg_temp_free(cmp1);

	}

	return;

    case 0x2007:		/* div0s Rm,Rn */

	{

	    gen_copy_bit_i32(cpu_sr, 8, REG(B11_8), 31);	/* SR_Q */

	    gen_copy_bit_i32(cpu_sr, 9, REG(B7_4), 31);		/* SR_M */

	    TCGv val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_xor_i32(val, REG(B7_4), REG(B11_8));

	    gen_copy_bit_i32(cpu_sr, 0, val, 31);		/* SR_T */

	    tcg_temp_free(val);

	}

	return;

    case 0x3004:		/* div1 Rm,Rn */

	tcg_gen_helper_1_2(helper_div1, REG(B11_8), REG(B7_4), REG(B11_8));

	return;

    case 0x300d:		/* dmuls.l Rm,Rn */

	{

	    TCGv tmp1 = tcg_temp_new(TCG_TYPE_I64);

	    TCGv tmp2 = tcg_temp_new(TCG_TYPE_I64);



	    tcg_gen_ext_i32_i64(tmp1, REG(B7_4));

	    tcg_gen_ext_i32_i64(tmp2, REG(B11_8));

	    tcg_gen_mul_i64(tmp1, tmp1, tmp2);

	    tcg_gen_trunc_i64_i32(cpu_macl, tmp1);

	    tcg_gen_shri_i64(tmp1, tmp1, 32);

	    tcg_gen_trunc_i64_i32(cpu_mach, tmp1);



	    tcg_temp_free(tmp2);

	    tcg_temp_free(tmp1);

	}

	return;

    case 0x3005:		/* dmulu.l Rm,Rn */

	{

	    TCGv tmp1 = tcg_temp_new(TCG_TYPE_I64);

	    TCGv tmp2 = tcg_temp_new(TCG_TYPE_I64);



	    tcg_gen_extu_i32_i64(tmp1, REG(B7_4));

	    tcg_gen_extu_i32_i64(tmp2, REG(B11_8));

	    tcg_gen_mul_i64(tmp1, tmp1, tmp2);

	    tcg_gen_trunc_i64_i32(cpu_macl, tmp1);

	    tcg_gen_shri_i64(tmp1, tmp1, 32);

	    tcg_gen_trunc_i64_i32(cpu_mach, tmp1);



	    tcg_temp_free(tmp2);

	    tcg_temp_free(tmp1);

	}

	return;

    case 0x600e:		/* exts.b Rm,Rn */

	tcg_gen_ext8s_i32(REG(B11_8), REG(B7_4));

	return;

    case 0x600f:		/* exts.w Rm,Rn */

	tcg_gen_ext16s_i32(REG(B11_8), REG(B7_4));

	return;

    case 0x600c:		/* extu.b Rm,Rn */

	tcg_gen_ext8u_i32(REG(B11_8), REG(B7_4));

	return;

    case 0x600d:		/* extu.w Rm,Rn */

	tcg_gen_ext16u_i32(REG(B11_8), REG(B7_4));

	return;

    case 0x000f:		/* mac.l @Rm+,@Rn+ */

	{

	    TCGv arg0, arg1;

	    arg0 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(arg0, REG(B7_4), ctx->memidx);

	    arg1 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(arg1, REG(B11_8), ctx->memidx);

	    tcg_gen_helper_0_2(helper_macl, arg0, arg1);

	    tcg_temp_free(arg1);

	    tcg_temp_free(arg0);

	    tcg_gen_addi_i32(REG(B7_4), REG(B7_4), 4);

	    tcg_gen_addi_i32(REG(B11_8), REG(B11_8), 4);

	}

	return;

    case 0x400f:		/* mac.w @Rm+,@Rn+ */

	{

	    TCGv arg0, arg1;

	    arg0 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(arg0, REG(B7_4), ctx->memidx);

	    arg1 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(arg1, REG(B11_8), ctx->memidx);

	    tcg_gen_helper_0_2(helper_macw, arg0, arg1);

	    tcg_temp_free(arg1);

	    tcg_temp_free(arg0);

	    tcg_gen_addi_i32(REG(B11_8), REG(B11_8), 2);

	    tcg_gen_addi_i32(REG(B7_4), REG(B7_4), 2);

	}

	return;

    case 0x0007:		/* mul.l Rm,Rn */

	tcg_gen_mul_i32(cpu_macl, REG(B7_4), REG(B11_8));

	return;

    case 0x200f:		/* muls.w Rm,Rn */

	{

	    TCGv arg0, arg1;

	    arg0 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_ext16s_i32(arg0, REG(B7_4));

	    arg1 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_ext16s_i32(arg1, REG(B11_8));

	    tcg_gen_mul_i32(cpu_macl, arg0, arg1);

	    tcg_temp_free(arg1);

	    tcg_temp_free(arg0);

	}

	return;

    case 0x200e:		/* mulu.w Rm,Rn */

	{

	    TCGv arg0, arg1;

	    arg0 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_ext16u_i32(arg0, REG(B7_4));

	    arg1 = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_ext16u_i32(arg1, REG(B11_8));

	    tcg_gen_mul_i32(cpu_macl, arg0, arg1);

	    tcg_temp_free(arg1);

	    tcg_temp_free(arg0);

	}

	return;

    case 0x600b:		/* neg Rm,Rn */

	tcg_gen_neg_i32(REG(B11_8), REG(B7_4));

	return;

    case 0x600a:		/* negc Rm,Rn */

	tcg_gen_helper_1_1(helper_negc, REG(B11_8), REG(B7_4));

	return;

    case 0x6007:		/* not Rm,Rn */

	tcg_gen_not_i32(REG(B11_8), REG(B7_4));

	return;

    case 0x200b:		/* or Rm,Rn */

	tcg_gen_or_i32(REG(B11_8), REG(B11_8), REG(B7_4));

	return;

    case 0x400c:		/* shad Rm,Rn */

	{

	    int label1 = gen_new_label();

	    int label2 = gen_new_label();

	    int label3 = gen_new_label();

	    int label4 = gen_new_label();

	    TCGv shift = tcg_temp_local_new(TCG_TYPE_I32);

	    tcg_gen_brcondi_i32(TCG_COND_LT, REG(B7_4), 0, label1);

	    /* Rm positive, shift to the left */

	    tcg_gen_andi_i32(shift, REG(B7_4), 0x1f);

	    tcg_gen_shl_i32(REG(B11_8), REG(B11_8), shift);

	    tcg_gen_br(label4);

	    /* Rm negative, shift to the right */

	    gen_set_label(label1);

	    tcg_gen_andi_i32(shift, REG(B7_4), 0x1f);

	    tcg_gen_brcondi_i32(TCG_COND_EQ, shift, 0, label2);

	    tcg_gen_not_i32(shift, REG(B7_4));

	    tcg_gen_andi_i32(shift, shift, 0x1f);

	    tcg_gen_addi_i32(shift, shift, 1);

	    tcg_gen_sar_i32(REG(B11_8), REG(B11_8), shift);

	    tcg_gen_br(label4);

	    /* Rm = -32 */

	    gen_set_label(label2);

	    tcg_gen_brcondi_i32(TCG_COND_LT, REG(B11_8), 0, label3);

	    tcg_gen_movi_i32(REG(B11_8), 0);

	    tcg_gen_br(label4);

	    gen_set_label(label3);

	    tcg_gen_movi_i32(REG(B11_8), 0xffffffff);

	    gen_set_label(label4);

	    tcg_temp_free(shift);

	}

	return;

    case 0x400d:		/* shld Rm,Rn */

	{

	    int label1 = gen_new_label();

	    int label2 = gen_new_label();

	    int label3 = gen_new_label();

	    TCGv shift = tcg_temp_local_new(TCG_TYPE_I32);

	    tcg_gen_brcondi_i32(TCG_COND_LT, REG(B7_4), 0, label1);

	    /* Rm positive, shift to the left */

	    tcg_gen_andi_i32(shift, REG(B7_4), 0x1f);

	    tcg_gen_shl_i32(REG(B11_8), REG(B11_8), shift);

	    tcg_gen_br(label3);

	    /* Rm negative, shift to the right */

	    gen_set_label(label1);

	    tcg_gen_andi_i32(shift, REG(B7_4), 0x1f);

	    tcg_gen_brcondi_i32(TCG_COND_EQ, shift, 0, label2);

	    tcg_gen_not_i32(shift, REG(B7_4));

	    tcg_gen_andi_i32(shift, shift, 0x1f);

	    tcg_gen_addi_i32(shift, shift, 1);

	    tcg_gen_shr_i32(REG(B11_8), REG(B11_8), shift);

	    tcg_gen_br(label3);

	    /* Rm = -32 */

	    gen_set_label(label2);

	    tcg_gen_movi_i32(REG(B11_8), 0);

	    gen_set_label(label3);

	    tcg_temp_free(shift);

	}

	return;

    case 0x3008:		/* sub Rm,Rn */

	tcg_gen_sub_i32(REG(B11_8), REG(B11_8), REG(B7_4));

	return;

    case 0x300a:		/* subc Rm,Rn */

	tcg_gen_helper_1_2(helper_subc, REG(B11_8), REG(B7_4), REG(B11_8));

	return;

    case 0x300b:		/* subv Rm,Rn */

	tcg_gen_helper_1_2(helper_subv, REG(B11_8), REG(B7_4), REG(B11_8));

	return;

    case 0x2008:		/* tst Rm,Rn */

	{

	    TCGv val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_and_i32(val, REG(B7_4), REG(B11_8));

	    gen_cmp_imm(TCG_COND_EQ, val, 0);

	    tcg_temp_free(val);

	}

	return;

    case 0x200a:		/* xor Rm,Rn */

	tcg_gen_xor_i32(REG(B11_8), REG(B11_8), REG(B7_4));

	return;

    case 0xf00c: /* fmov {F,D,X}Rm,{F,D,X}Rn - FPSCR: Nothing */

	if (ctx->fpscr & FPSCR_SZ) {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    gen_load_fpr64(fp, XREG(B7_4));

	    gen_store_fpr64(fp, XREG(B11_8));

	    tcg_temp_free(fp);

	} else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B7_4));

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf00a: /* fmov {F,D,X}Rm,@Rn - FPSCR: Nothing */

	if (ctx->fpscr & FPSCR_SZ) {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    gen_load_fpr64(fp, XREG(B7_4));

	    tcg_gen_qemu_st64(fp, REG(B11_8), ctx->memidx);

	    tcg_temp_free(fp);

	} else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B7_4));

	    tcg_gen_qemu_st32(fp, REG(B11_8), ctx->memidx);

	    tcg_temp_free(fp);

	}

	return;

    case 0xf008: /* fmov @Rm,{F,D,X}Rn - FPSCR: Nothing */

	if (ctx->fpscr & FPSCR_SZ) {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    tcg_gen_qemu_ld64(fp, REG(B7_4), ctx->memidx);

	    gen_store_fpr64(fp, XREG(B11_8));

	    tcg_temp_free(fp);

	} else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32u(fp, REG(B7_4), ctx->memidx);

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf009: /* fmov @Rm+,{F,D,X}Rn - FPSCR: Nothing */

	if (ctx->fpscr & FPSCR_SZ) {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    tcg_gen_qemu_ld64(fp, REG(B7_4), ctx->memidx);

	    gen_store_fpr64(fp, XREG(B11_8));

	    tcg_temp_free(fp);

	    tcg_gen_addi_i32(REG(B7_4),REG(B7_4), 8);

	} else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32u(fp, REG(B7_4), ctx->memidx);

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	    tcg_gen_addi_i32(REG(B7_4), REG(B7_4), 4);

	}

	return;

    case 0xf00b: /* fmov {F,D,X}Rm,@-Rn - FPSCR: Nothing */

	if (ctx->fpscr & FPSCR_SZ) {

	    TCGv addr, fp;

	    addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 8);

	    fp = tcg_temp_new(TCG_TYPE_I64);

	    gen_load_fpr64(fp, XREG(B7_4));

	    tcg_gen_qemu_st64(fp, addr, ctx->memidx);

	    tcg_temp_free(fp);

	    tcg_temp_free(addr);

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 8);

	} else {

	    TCGv addr, fp;

	    addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 4);

	    fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B7_4));

	    tcg_gen_qemu_st32(fp, addr, ctx->memidx);

	    tcg_temp_free(fp);

	    tcg_temp_free(addr);

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 4);

	}

	return;

    case 0xf006: /* fmov @(R0,Rm),{F,D,X}Rm - FPSCR: Nothing */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B7_4), REG(0));

	    if (ctx->fpscr & FPSCR_SZ) {

		TCGv fp = tcg_temp_new(TCG_TYPE_I64);

		tcg_gen_qemu_ld64(fp, addr, ctx->memidx);

		gen_store_fpr64(fp, XREG(B11_8));

		tcg_temp_free(fp);

	    } else {

		TCGv fp = tcg_temp_new(TCG_TYPE_I32);

		tcg_gen_qemu_ld32u(fp, addr, ctx->memidx);

		gen_store_fpr32(fp, FREG(B11_8));

		tcg_temp_free(fp);

	    }

	    tcg_temp_free(addr);

	}

	return;

    case 0xf007: /* fmov {F,D,X}Rn,@(R0,Rn) - FPSCR: Nothing */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(B11_8), REG(0));

	    if (ctx->fpscr & FPSCR_SZ) {

		TCGv fp = tcg_temp_new(TCG_TYPE_I64);

		gen_load_fpr64(fp, XREG(B7_4));

		tcg_gen_qemu_st64(fp, addr, ctx->memidx);

		tcg_temp_free(fp);

	    } else {

		TCGv fp = tcg_temp_new(TCG_TYPE_I32);

		gen_load_fpr32(fp, FREG(B7_4));

		tcg_gen_qemu_st32(fp, addr, ctx->memidx);

		tcg_temp_free(fp);

	    }

	    tcg_temp_free(addr);

	}

	return;

    case 0xf000: /* fadd Rm,Rn - FPSCR: R[PR,Enable.O/U/I]/W[Cause,Flag] */

    case 0xf001: /* fsub Rm,Rn - FPSCR: R[PR,Enable.O/U/I]/W[Cause,Flag] */

    case 0xf002: /* fmul Rm,Rn - FPSCR: R[PR,Enable.O/U/I]/W[Cause,Flag] */

    case 0xf003: /* fdiv Rm,Rn - FPSCR: R[PR,Enable.O/U/I]/W[Cause,Flag] */

    case 0xf004: /* fcmp/eq Rm,Rn - FPSCR: R[PR,Enable.V]/W[Cause,Flag] */

    case 0xf005: /* fcmp/gt Rm,Rn - FPSCR: R[PR,Enable.V]/W[Cause,Flag] */

	{

	    TCGv fp0, fp1;



	    if (ctx->fpscr & FPSCR_PR) {

		if (ctx->opcode & 0x0110)

		    break; /* illegal instruction */

		fp0 = tcg_temp_new(TCG_TYPE_I64);

		fp1 = tcg_temp_new(TCG_TYPE_I64);

		gen_load_fpr64(fp0, DREG(B11_8));

		gen_load_fpr64(fp1, DREG(B7_4));

	    }

	    else {

		fp0 = tcg_temp_new(TCG_TYPE_I32);

		fp1 = tcg_temp_new(TCG_TYPE_I32);

		gen_load_fpr32(fp0, FREG(B11_8));

		gen_load_fpr32(fp1, FREG(B7_4));

	    }



	    switch (ctx->opcode & 0xf00f) {

	    case 0xf000:		/* fadd Rm,Rn */

		if (ctx->fpscr & FPSCR_PR)

		    tcg_gen_helper_1_2(helper_fadd_DT, fp0, fp0, fp1);

		else

		    tcg_gen_helper_1_2(helper_fadd_FT, fp0, fp0, fp1);

		break;

	    case 0xf001:		/* fsub Rm,Rn */

		if (ctx->fpscr & FPSCR_PR)

		    tcg_gen_helper_1_2(helper_fsub_DT, fp0, fp0, fp1);

		else

		    tcg_gen_helper_1_2(helper_fsub_FT, fp0, fp0, fp1);

		break;

	    case 0xf002:		/* fmul Rm,Rn */

		if (ctx->fpscr & FPSCR_PR)

		    tcg_gen_helper_1_2(helper_fmul_DT, fp0, fp0, fp1);

		else

		    tcg_gen_helper_1_2(helper_fmul_FT, fp0, fp0, fp1);

		break;

	    case 0xf003:		/* fdiv Rm,Rn */

		if (ctx->fpscr & FPSCR_PR)

		    tcg_gen_helper_1_2(helper_fdiv_DT, fp0, fp0, fp1);

		else

		    tcg_gen_helper_1_2(helper_fdiv_FT, fp0, fp0, fp1);

		break;

	    case 0xf004:		/* fcmp/eq Rm,Rn */

		if (ctx->fpscr & FPSCR_PR)

		    tcg_gen_helper_0_2(helper_fcmp_eq_DT, fp0, fp1);

		else

		    tcg_gen_helper_0_2(helper_fcmp_eq_FT, fp0, fp1);

		return;

	    case 0xf005:		/* fcmp/gt Rm,Rn */

		if (ctx->fpscr & FPSCR_PR)

		    tcg_gen_helper_0_2(helper_fcmp_gt_DT, fp0, fp1);

		else

		    tcg_gen_helper_0_2(helper_fcmp_gt_FT, fp0, fp1);

		return;

	    }



	    if (ctx->fpscr & FPSCR_PR) {

		gen_store_fpr64(fp0, DREG(B11_8));

	    }

	    else {

		gen_store_fpr32(fp0, FREG(B11_8));

	    }

	    tcg_temp_free(fp1);

	    tcg_temp_free(fp0);

	}

	return;

    }



    switch (ctx->opcode & 0xff00) {

    case 0xc900:		/* and #imm,R0 */

	tcg_gen_andi_i32(REG(0), REG(0), B7_0);

	return;

    case 0xcd00:		/* and.b #imm,@(R0,GBR) */

	{

	    TCGv addr, val;

	    addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(0), cpu_gbr);

	    val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld8u(val, addr, ctx->memidx);

	    tcg_gen_andi_i32(val, val, B7_0);

	    tcg_gen_qemu_st8(val, addr, ctx->memidx);

	    tcg_temp_free(val);

	    tcg_temp_free(addr);

	}

	return;

    case 0x8b00:		/* bf label */

	CHECK_NOT_DELAY_SLOT

	    gen_conditional_jump(ctx, ctx->pc + 2,

				 ctx->pc + 4 + B7_0s * 2);

	ctx->bstate = BS_BRANCH;

	return;

    case 0x8f00:		/* bf/s label */

	CHECK_NOT_DELAY_SLOT

	gen_branch_slot(ctx->delayed_pc = ctx->pc + 4 + B7_0s * 2, 0);

	ctx->flags |= DELAY_SLOT_CONDITIONAL;

	return;

    case 0x8900:		/* bt label */

	CHECK_NOT_DELAY_SLOT

	    gen_conditional_jump(ctx, ctx->pc + 4 + B7_0s * 2,

				 ctx->pc + 2);

	ctx->bstate = BS_BRANCH;

	return;

    case 0x8d00:		/* bt/s label */

	CHECK_NOT_DELAY_SLOT

	gen_branch_slot(ctx->delayed_pc = ctx->pc + 4 + B7_0s * 2, 1);

	ctx->flags |= DELAY_SLOT_CONDITIONAL;

	return;

    case 0x8800:		/* cmp/eq #imm,R0 */

	gen_cmp_imm(TCG_COND_EQ, REG(0), B7_0s);

	return;

    case 0xc400:		/* mov.b @(disp,GBR),R0 */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, cpu_gbr, B7_0);

	    tcg_gen_qemu_ld8s(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xc500:		/* mov.w @(disp,GBR),R0 */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, cpu_gbr, B7_0 * 2);

	    tcg_gen_qemu_ld16s(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xc600:		/* mov.l @(disp,GBR),R0 */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, cpu_gbr, B7_0 * 4);

	    tcg_gen_qemu_ld32s(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xc000:		/* mov.b R0,@(disp,GBR) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, cpu_gbr, B7_0);

	    tcg_gen_qemu_st8(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xc100:		/* mov.w R0,@(disp,GBR) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, cpu_gbr, B7_0 * 2);

	    tcg_gen_qemu_st16(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xc200:		/* mov.l R0,@(disp,GBR) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, cpu_gbr, B7_0 * 4);

	    tcg_gen_qemu_st32(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x8000:		/* mov.b R0,@(disp,Rn) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, REG(B7_4), B3_0);

	    tcg_gen_qemu_st8(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x8100:		/* mov.w R0,@(disp,Rn) */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, REG(B7_4), B3_0 * 2);

	    tcg_gen_qemu_st16(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x8400:		/* mov.b @(disp,Rn),R0 */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, REG(B7_4), B3_0);

	    tcg_gen_qemu_ld8s(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0x8500:		/* mov.w @(disp,Rn),R0 */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_addi_i32(addr, REG(B7_4), B3_0 * 2);

	    tcg_gen_qemu_ld16s(REG(0), addr, ctx->memidx);

	    tcg_temp_free(addr);

	}

	return;

    case 0xc700:		/* mova @(disp,PC),R0 */

	tcg_gen_movi_i32(REG(0), ((ctx->pc & 0xfffffffc) + 4 + B7_0 * 4) & ~3);

	return;

    case 0xcb00:		/* or #imm,R0 */

	tcg_gen_ori_i32(REG(0), REG(0), B7_0);

	return;

    case 0xcf00:		/* or.b #imm,@(R0,GBR) */

	{

	    TCGv addr, val;

	    addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(0), cpu_gbr);

	    val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld8u(val, addr, ctx->memidx);

	    tcg_gen_ori_i32(val, val, B7_0);

	    tcg_gen_qemu_st8(val, addr, ctx->memidx);

	    tcg_temp_free(val);

	    tcg_temp_free(addr);

	}

	return;

    case 0xc300:		/* trapa #imm */

	{

	    TCGv imm;

	    CHECK_NOT_DELAY_SLOT

	    tcg_gen_movi_i32(cpu_pc, ctx->pc);

	    imm = tcg_const_i32(B7_0);

	    tcg_gen_helper_0_1(helper_trapa, imm);

	    tcg_temp_free(imm);

	    ctx->bstate = BS_BRANCH;

	}

	return;

    case 0xc800:		/* tst #imm,R0 */

	{

	    TCGv val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_andi_i32(val, REG(0), B7_0);

	    gen_cmp_imm(TCG_COND_EQ, val, 0);

	    tcg_temp_free(val);

	}

	return;

    case 0xcc00:		/* tst.b #imm,@(R0,GBR) */

	{

	    TCGv val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(val, REG(0), cpu_gbr);

	    tcg_gen_qemu_ld8u(val, val, ctx->memidx);

	    tcg_gen_andi_i32(val, val, B7_0);

	    gen_cmp_imm(TCG_COND_EQ, val, 0);

	    tcg_temp_free(val);

	}

	return;

    case 0xca00:		/* xor #imm,R0 */

	tcg_gen_xori_i32(REG(0), REG(0), B7_0);

	return;

    case 0xce00:		/* xor.b #imm,@(R0,GBR) */

	{

	    TCGv addr, val;

	    addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_add_i32(addr, REG(0), cpu_gbr);

	    val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld8u(val, addr, ctx->memidx);

	    tcg_gen_xori_i32(val, val, B7_0);

	    tcg_gen_qemu_st8(val, addr, ctx->memidx);

	    tcg_temp_free(val);

	    tcg_temp_free(addr);

	}

	return;

    }



    switch (ctx->opcode & 0xf08f) {

    case 0x408e:		/* ldc Rm,Rn_BANK */

	tcg_gen_mov_i32(ALTREG(B6_4), REG(B11_8));

	return;

    case 0x4087:		/* ldc.l @Rm+,Rn_BANK */

	tcg_gen_qemu_ld32s(ALTREG(B6_4), REG(B11_8), ctx->memidx);

	tcg_gen_addi_i32(REG(B11_8), REG(B11_8), 4);

	return;

    case 0x0082:		/* stc Rm_BANK,Rn */

	tcg_gen_mov_i32(REG(B11_8), ALTREG(B6_4));

	return;

    case 0x4083:		/* stc.l Rm_BANK,@-Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 4);

	    tcg_gen_qemu_st32(ALTREG(B6_4), addr, ctx->memidx);

	    tcg_temp_free(addr);

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 4);

	}

	return;

    }



    switch (ctx->opcode & 0xf0ff) {

    case 0x0023:		/* braf Rn */

	CHECK_NOT_DELAY_SLOT

	tcg_gen_addi_i32(cpu_delayed_pc, REG(B11_8), ctx->pc + 4);

	ctx->flags |= DELAY_SLOT;

	ctx->delayed_pc = (uint32_t) - 1;

	return;

    case 0x0003:		/* bsrf Rn */

	CHECK_NOT_DELAY_SLOT

	tcg_gen_movi_i32(cpu_pr, ctx->pc + 4);

	tcg_gen_add_i32(cpu_delayed_pc, REG(B11_8), cpu_pr);

	ctx->flags |= DELAY_SLOT;

	ctx->delayed_pc = (uint32_t) - 1;

	return;

    case 0x4015:		/* cmp/pl Rn */

	gen_cmp_imm(TCG_COND_GT, REG(B11_8), 0);

	return;

    case 0x4011:		/* cmp/pz Rn */

	gen_cmp_imm(TCG_COND_GE, REG(B11_8), 0);

	return;

    case 0x4010:		/* dt Rn */

	tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 1);

	gen_cmp_imm(TCG_COND_EQ, REG(B11_8), 0);

	return;

    case 0x402b:		/* jmp @Rn */

	CHECK_NOT_DELAY_SLOT

	tcg_gen_mov_i32(cpu_delayed_pc, REG(B11_8));

	ctx->flags |= DELAY_SLOT;

	ctx->delayed_pc = (uint32_t) - 1;

	return;

    case 0x400b:		/* jsr @Rn */

	CHECK_NOT_DELAY_SLOT

	tcg_gen_movi_i32(cpu_pr, ctx->pc + 4);

	tcg_gen_mov_i32(cpu_delayed_pc, REG(B11_8));

	ctx->flags |= DELAY_SLOT;

	ctx->delayed_pc = (uint32_t) - 1;

	return;

    case 0x400e:		/* lds Rm,SR */

	tcg_gen_andi_i32(cpu_sr, REG(B11_8), 0x700083f3);

	ctx->bstate = BS_STOP;

	return;

    case 0x4007:		/* lds.l @Rm+,SR */

	{

	    TCGv val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(val, REG(B11_8), ctx->memidx);

	    tcg_gen_andi_i32(cpu_sr, val, 0x700083f3);

	    tcg_temp_free(val);

	    tcg_gen_addi_i32(REG(B11_8), REG(B11_8), 4);

	    ctx->bstate = BS_STOP;

	}

	return;

    case 0x0002:		/* sts SR,Rn */

	tcg_gen_mov_i32(REG(B11_8), cpu_sr);

	return;

    case 0x4003:		/* sts SR,@-Rn */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 4);

	    tcg_gen_qemu_st32(cpu_sr, addr, ctx->memidx);

	    tcg_temp_free(addr);

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 4);

	}

	return;

#define LDST(reg,ldnum,ldpnum,stnum,stpnum)			\

  case ldnum:							\

    tcg_gen_mov_i32 (cpu_##reg, REG(B11_8));			\

    return;							\

  case ldpnum:							\

    tcg_gen_qemu_ld32s (cpu_##reg, REG(B11_8), ctx->memidx);	\

    tcg_gen_addi_i32(REG(B11_8), REG(B11_8), 4);		\

    return;							\

  case stnum:							\

    tcg_gen_mov_i32 (REG(B11_8), cpu_##reg);			\

    return;							\

  case stpnum:							\

    {								\

	TCGv addr = tcg_temp_new(TCG_TYPE_I32);			\

	tcg_gen_subi_i32(addr, REG(B11_8), 4);			\

	tcg_gen_qemu_st32 (cpu_##reg, addr, ctx->memidx);	\

	tcg_temp_free(addr);					\

	tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 4);		\

    }								\

    return;

	LDST(gbr,  0x401e, 0x4017, 0x0012, 0x4013)

	LDST(vbr,  0x402e, 0x4027, 0x0022, 0x4023)

	LDST(ssr,  0x403e, 0x4037, 0x0032, 0x4033)

	LDST(spc,  0x404e, 0x4047, 0x0042, 0x4043)

	LDST(dbr,  0x40fa, 0x40f6, 0x00fa, 0x40f2)

	LDST(mach, 0x400a, 0x4006, 0x000a, 0x4002)

	LDST(macl, 0x401a, 0x4016, 0x001a, 0x4012)

	LDST(pr,   0x402a, 0x4026, 0x002a, 0x4022)

	LDST(fpul, 0x405a, 0x4056, 0x005a, 0x4052)

    case 0x406a:		/* lds Rm,FPSCR */

	tcg_gen_helper_0_1(helper_ld_fpscr, REG(B11_8));

	ctx->bstate = BS_STOP;

	return;

    case 0x4066:		/* lds.l @Rm+,FPSCR */

	{

	    TCGv addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(addr, REG(B11_8), ctx->memidx);

	    tcg_gen_addi_i32(REG(B11_8), REG(B11_8), 4);

	    tcg_gen_helper_0_1(helper_ld_fpscr, addr);

	    tcg_temp_free(addr);

	    ctx->bstate = BS_STOP;

	}

	return;

    case 0x006a:		/* sts FPSCR,Rn */

	tcg_gen_andi_i32(REG(B11_8), cpu_fpscr, 0x003fffff);

	return;

    case 0x4062:		/* sts FPSCR,@-Rn */

	{

	    TCGv addr, val;

	    val = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_andi_i32(val, cpu_fpscr, 0x003fffff);

	    addr = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_subi_i32(addr, REG(B11_8), 4);

	    tcg_gen_qemu_st32(val, addr, ctx->memidx);

	    tcg_temp_free(addr);

	    tcg_temp_free(val);

	    tcg_gen_subi_i32(REG(B11_8), REG(B11_8), 4);

	}

	return;

    case 0x00c3:		/* movca.l R0,@Rm */

	tcg_gen_qemu_st32(REG(0), REG(B11_8), ctx->memidx);

	return;

    case 0x0029:		/* movt Rn */

	tcg_gen_andi_i32(REG(B11_8), cpu_sr, SR_T);

	return;

    case 0x0093:		/* ocbi @Rn */

	{

	    TCGv dummy = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(dummy, REG(B11_8), ctx->memidx);

	    tcg_temp_free(dummy);

	}

	return;

    case 0x00a3:		/* ocbp @Rn */

	{

	    TCGv dummy = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(dummy, REG(B11_8), ctx->memidx);

	    tcg_temp_free(dummy);

	}

	return;

    case 0x00b3:		/* ocbwb @Rn */

	{

	    TCGv dummy = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld32s(dummy, REG(B11_8), ctx->memidx);

	    tcg_temp_free(dummy);

	}

	return;

    case 0x0083:		/* pref @Rn */

	return;

    case 0x4024:		/* rotcl Rn */

	{

	    TCGv tmp = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_mov_i32(tmp, cpu_sr);

	    gen_copy_bit_i32(cpu_sr, 0, REG(B11_8), 31);

	    tcg_gen_shli_i32(REG(B11_8), REG(B11_8), 1);

	    gen_copy_bit_i32(REG(B11_8), 0, tmp, 0);

	    tcg_temp_free(tmp);

	}

	return;

    case 0x4025:		/* rotcr Rn */

	{

	    TCGv tmp = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_mov_i32(tmp, cpu_sr);

	    gen_copy_bit_i32(cpu_sr, 0, REG(B11_8), 0);

	    tcg_gen_shri_i32(REG(B11_8), REG(B11_8), 1);

	    gen_copy_bit_i32(REG(B11_8), 31, tmp, 0);

	    tcg_temp_free(tmp);

	}

	return;

    case 0x4004:		/* rotl Rn */

	gen_copy_bit_i32(cpu_sr, 0, REG(B11_8), 31);

	tcg_gen_shli_i32(REG(B11_8), REG(B11_8), 1);

	gen_copy_bit_i32(REG(B11_8), 0, cpu_sr, 0);

	return;

    case 0x4005:		/* rotr Rn */

	gen_copy_bit_i32(cpu_sr, 0, REG(B11_8), 0);

	tcg_gen_shri_i32(REG(B11_8), REG(B11_8), 1);

	gen_copy_bit_i32(REG(B11_8), 31, cpu_sr, 0);

	return;

    case 0x4000:		/* shll Rn */

    case 0x4020:		/* shal Rn */

	gen_copy_bit_i32(cpu_sr, 0, REG(B11_8), 31);

	tcg_gen_shli_i32(REG(B11_8), REG(B11_8), 1);

	return;

    case 0x4021:		/* shar Rn */

	gen_copy_bit_i32(cpu_sr, 0, REG(B11_8), 0);

	tcg_gen_sari_i32(REG(B11_8), REG(B11_8), 1);

	return;

    case 0x4001:		/* shlr Rn */

	gen_copy_bit_i32(cpu_sr, 0, REG(B11_8), 0);

	tcg_gen_shri_i32(REG(B11_8), REG(B11_8), 1);

	return;

    case 0x4008:		/* shll2 Rn */

	tcg_gen_shli_i32(REG(B11_8), REG(B11_8), 2);

	return;

    case 0x4018:		/* shll8 Rn */

	tcg_gen_shli_i32(REG(B11_8), REG(B11_8), 8);

	return;

    case 0x4028:		/* shll16 Rn */

	tcg_gen_shli_i32(REG(B11_8), REG(B11_8), 16);

	return;

    case 0x4009:		/* shlr2 Rn */

	tcg_gen_shri_i32(REG(B11_8), REG(B11_8), 2);

	return;

    case 0x4019:		/* shlr8 Rn */

	tcg_gen_shri_i32(REG(B11_8), REG(B11_8), 8);

	return;

    case 0x4029:		/* shlr16 Rn */

	tcg_gen_shri_i32(REG(B11_8), REG(B11_8), 16);

	return;

    case 0x401b:		/* tas.b @Rn */

	{

	    TCGv addr, val;

	    addr = tcg_temp_local_new(TCG_TYPE_I32);

	    tcg_gen_mov_i32(addr, REG(B11_8));

	    val = tcg_temp_local_new(TCG_TYPE_I32);

	    tcg_gen_qemu_ld8u(val, addr, ctx->memidx);

	    gen_cmp_imm(TCG_COND_EQ, val, 0);

	    tcg_gen_ori_i32(val, val, 0x80);

	    tcg_gen_qemu_st8(val, addr, ctx->memidx);

	    tcg_temp_free(val);

	    tcg_temp_free(addr);

	}

	return;

    case 0xf00d: /* fsts FPUL,FRn - FPSCR: Nothing */

	{

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_mov_i32(fp, cpu_fpul);

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf01d: /* flds FRm,FPUL - FPSCR: Nothing */

	{

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B11_8));

	    tcg_gen_mov_i32(cpu_fpul, fp);

	    tcg_temp_free(fp);

	}

	return;

    case 0xf02d: /* float FPUL,FRn/DRn - FPSCR: R[PR,Enable.I]/W[Cause,Flag] */

	if (ctx->fpscr & FPSCR_PR) {

	    TCGv fp;

	    if (ctx->opcode & 0x0100)

		break; /* illegal instruction */

	    fp = tcg_temp_new(TCG_TYPE_I64);

	    tcg_gen_helper_1_1(helper_float_DT, fp, cpu_fpul);

	    gen_store_fpr64(fp, DREG(B11_8));

	    tcg_temp_free(fp);

	}

	else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    tcg_gen_helper_1_1(helper_float_FT, fp, cpu_fpul);

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf03d: /* ftrc FRm/DRm,FPUL - FPSCR: R[PR,Enable.V]/W[Cause,Flag] */

	if (ctx->fpscr & FPSCR_PR) {

	    TCGv fp;

	    if (ctx->opcode & 0x0100)

		break; /* illegal instruction */

	    fp = tcg_temp_new(TCG_TYPE_I64);

	    gen_load_fpr64(fp, DREG(B11_8));

	    tcg_gen_helper_1_1(helper_ftrc_DT, cpu_fpul, fp);

	    tcg_temp_free(fp);

	}

	else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B11_8));

	    tcg_gen_helper_1_1(helper_ftrc_FT, cpu_fpul, fp);

	    tcg_temp_free(fp);

	}

	return;

    case 0xf04d: /* fneg FRn/DRn - FPSCR: Nothing */

	{

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B11_8));

	    tcg_gen_helper_1_1(helper_fneg_T, fp, fp);

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf05d: /* fabs FRn/DRn */

	if (ctx->fpscr & FPSCR_PR) {

	    if (ctx->opcode & 0x0100)

		break; /* illegal instruction */

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    gen_load_fpr64(fp, DREG(B11_8));

	    tcg_gen_helper_1_1(helper_fabs_DT, fp, fp);

	    gen_store_fpr64(fp, DREG(B11_8));

	    tcg_temp_free(fp);

	} else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B11_8));

	    tcg_gen_helper_1_1(helper_fabs_FT, fp, fp);

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf06d: /* fsqrt FRn */

	if (ctx->fpscr & FPSCR_PR) {

	    if (ctx->opcode & 0x0100)

		break; /* illegal instruction */

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    gen_load_fpr64(fp, DREG(B11_8));

	    tcg_gen_helper_1_1(helper_fsqrt_DT, fp, fp);

	    gen_store_fpr64(fp, DREG(B11_8));

	    tcg_temp_free(fp);

	} else {

	    TCGv fp = tcg_temp_new(TCG_TYPE_I32);

	    gen_load_fpr32(fp, FREG(B11_8));

	    tcg_gen_helper_1_1(helper_fsqrt_FT, fp, fp);

	    gen_store_fpr32(fp, FREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf07d: /* fsrra FRn */

	break;

    case 0xf08d: /* fldi0 FRn - FPSCR: R[PR] */

	if (!(ctx->fpscr & FPSCR_PR)) {

	    TCGv val = tcg_const_i32(0);

	    gen_load_fpr32(val, FREG(B11_8));

	    tcg_temp_free(val);

	    return;

	}

	break;

    case 0xf09d: /* fldi1 FRn - FPSCR: R[PR] */

	if (!(ctx->fpscr & FPSCR_PR)) {

	    TCGv val = tcg_const_i32(0x3f800000);

	    gen_load_fpr32(val, FREG(B11_8));

	    tcg_temp_free(val);

	    return;

	}

	break;

    case 0xf0ad: /* fcnvsd FPUL,DRn */

	{

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    tcg_gen_helper_1_1(helper_fcnvsd_FT_DT, fp, cpu_fpul);

	    gen_store_fpr64(fp, DREG(B11_8));

	    tcg_temp_free(fp);

	}

	return;

    case 0xf0bd: /* fcnvds DRn,FPUL */

	{

	    TCGv fp = tcg_temp_new(TCG_TYPE_I64);

	    gen_load_fpr64(fp, DREG(B11_8));

	    tcg_gen_helper_1_1(helper_fcnvds_DT_FT, cpu_fpul, fp);

	    tcg_temp_free(fp);

	}

	return;

    }



    fprintf(stderr, "unknown instruction 0x%04x at pc 0x%08x\n",

	    ctx->opcode, ctx->pc);

    tcg_gen_helper_0_0(helper_raise_illegal_instruction);

    ctx->bstate = BS_EXCP;

}
