static void gen_load_fp(DisasContext *s, int opsize, TCGv addr, TCGv_ptr fp)

{

    TCGv tmp;

    TCGv_i64 t64;

    int index = IS_USER(s);



    t64 = tcg_temp_new_i64();

    tmp = tcg_temp_new();

    switch (opsize) {

    case OS_BYTE:

        tcg_gen_qemu_ld8s(tmp, addr, index);

        gen_helper_exts32(cpu_env, fp, tmp);

        break;

    case OS_WORD:

        tcg_gen_qemu_ld16s(tmp, addr, index);

        gen_helper_exts32(cpu_env, fp, tmp);

        break;

    case OS_LONG:

        tcg_gen_qemu_ld32u(tmp, addr, index);

        gen_helper_exts32(cpu_env, fp, tmp);

        break;

    case OS_SINGLE:

        tcg_gen_qemu_ld32u(tmp, addr, index);

        gen_helper_extf32(cpu_env, fp, tmp);

        break;

    case OS_DOUBLE:

        tcg_gen_qemu_ld64(t64, addr, index);

        gen_helper_extf64(cpu_env, fp, t64);

        tcg_temp_free_i64(t64);

        break;

    case OS_EXTENDED:

        if (m68k_feature(s->env, M68K_FEATURE_CF_FPU)) {

            gen_exception(s, s->insn_pc, EXCP_FP_UNIMP);

            break;

        }

        tcg_gen_qemu_ld32u(tmp, addr, index);

        tcg_gen_shri_i32(tmp, tmp, 16);

        tcg_gen_st16_i32(tmp, fp, offsetof(FPReg, l.upper));

        tcg_gen_addi_i32(tmp, addr, 4);

        tcg_gen_qemu_ld64(t64, tmp, index);

        tcg_gen_st_i64(t64, fp, offsetof(FPReg, l.lower));

        break;

    case OS_PACKED:

        /* unimplemented data type on 68040/ColdFire

         * FIXME if needed for another FPU

         */

        gen_exception(s, s->insn_pc, EXCP_FP_UNIMP);

        break;

    default:

        g_assert_not_reached();

    }

    tcg_temp_free(tmp);

    tcg_temp_free_i64(t64);

}
