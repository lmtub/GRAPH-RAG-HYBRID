static void gen_flt3_ldst (DisasContext *ctx, uint32_t opc,

                           int fd, int fs, int base, int index)

{

    const char *opn = "extended float load/store";

    int store = 0;

    TCGv t0 = tcg_temp_new();



    if (base == 0) {

        gen_load_gpr(t0, index);

    } else if (index == 0) {

        gen_load_gpr(t0, base);

    } else {

        gen_load_gpr(t0, index);

        gen_op_addr_add(ctx, t0, cpu_gpr[base]);

    }

    /* Don't do NOP if destination is zero: we must perform the actual

       memory access. */

    save_cpu_state(ctx, 0);

    switch (opc) {

    case OPC_LWXC1:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            tcg_gen_qemu_ld32s(t0, t0, ctx->mem_idx);

            tcg_gen_trunc_tl_i32(fp0, t0);

            gen_store_fpr32(fp0, fd);

            tcg_temp_free_i32(fp0);

        }

        opn = "lwxc1";

        break;

    case OPC_LDXC1:

        check_cop1x(ctx);

        check_cp1_registers(ctx, fd);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            tcg_gen_qemu_ld64(fp0, t0, ctx->mem_idx);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "ldxc1";

        break;

    case OPC_LUXC1:

        check_cp1_64bitmode(ctx);

        tcg_gen_andi_tl(t0, t0, ~0x7);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            tcg_gen_qemu_ld64(fp0, t0, ctx->mem_idx);

            gen_store_fpr64(ctx, fp0, fd);

            tcg_temp_free_i64(fp0);

        }

        opn = "luxc1";

        break;

    case OPC_SWXC1:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv t1 = tcg_temp_new();



            gen_load_fpr32(fp0, fs);

            tcg_gen_extu_i32_tl(t1, fp0);

            tcg_gen_qemu_st32(t1, t0, ctx->mem_idx);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(t1);

        }

        opn = "swxc1";

        store = 1;

        break;

    case OPC_SDXC1:

        check_cop1x(ctx);

        check_cp1_registers(ctx, fs);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            tcg_gen_qemu_st64(fp0, t0, ctx->mem_idx);

            tcg_temp_free_i64(fp0);

        }

        opn = "sdxc1";

        store = 1;

        break;

    case OPC_SUXC1:

        check_cp1_64bitmode(ctx);

        tcg_gen_andi_tl(t0, t0, ~0x7);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            tcg_gen_qemu_st64(fp0, t0, ctx->mem_idx);

            tcg_temp_free_i64(fp0);

        }

        opn = "suxc1";

        store = 1;

        break;

    }

    tcg_temp_free(t0);

    MIPS_DEBUG("%s %s, %s(%s)", opn, fregnames[store ? fs : fd],

               regnames[index], regnames[base]);

}
