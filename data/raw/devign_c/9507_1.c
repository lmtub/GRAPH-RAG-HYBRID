static int disas_dsp_insn(CPUState *env, DisasContext *s, uint32_t insn)

{

    int acc, rd0, rd1, rdhi, rdlo;

    TCGv tmp, tmp2;



    if ((insn & 0x0ff00f10) == 0x0e200010) {

        /* Multiply with Internal Accumulate Format */

        rd0 = (insn >> 12) & 0xf;

        rd1 = insn & 0xf;

        acc = (insn >> 5) & 7;



        if (acc != 0)

            return 1;



        tmp = load_reg(s, rd0);

        tmp2 = load_reg(s, rd1);

        switch ((insn >> 16) & 0xf) {

        case 0x0:					/* MIA */

            gen_helper_iwmmxt_muladdsl(cpu_M0, cpu_M0, tmp, tmp2);

            break;

        case 0x8:					/* MIAPH */

            gen_helper_iwmmxt_muladdsw(cpu_M0, cpu_M0, tmp, tmp2);

            break;

        case 0xc:					/* MIABB */

        case 0xd:					/* MIABT */

        case 0xe:					/* MIATB */

        case 0xf:					/* MIATT */

            if (insn & (1 << 16))

                tcg_gen_shri_i32(tmp, tmp, 16);

            if (insn & (1 << 17))

                tcg_gen_shri_i32(tmp2, tmp2, 16);

            gen_helper_iwmmxt_muladdswl(cpu_M0, cpu_M0, tmp, tmp2);

            break;

        default:

            return 1;

        }

        dead_tmp(tmp2);

        dead_tmp(tmp);



        gen_op_iwmmxt_movq_wRn_M0(acc);

        return 0;

    }



    if ((insn & 0x0fe00ff8) == 0x0c400000) {

        /* Internal Accumulator Access Format */

        rdhi = (insn >> 16) & 0xf;

        rdlo = (insn >> 12) & 0xf;

        acc = insn & 7;



        if (acc != 0)

            return 1;



        if (insn & ARM_CP_RW_BIT) {			/* MRA */

            iwmmxt_load_reg(cpu_V0, acc);

            tcg_gen_trunc_i64_i32(cpu_R[rdlo], cpu_V0);

            tcg_gen_shri_i64(cpu_V0, cpu_V0, 32);

            tcg_gen_trunc_i64_i32(cpu_R[rdhi], cpu_V0);

            tcg_gen_andi_i32(cpu_R[rdhi], cpu_R[rdhi], (1 << (40 - 32)) - 1);

        } else {					/* MAR */

            tcg_gen_concat_i32_i64(cpu_V0, cpu_R[rdlo], cpu_R[rdhi]);

            iwmmxt_store_reg(cpu_V0, acc);

        }

        return 0;

    }



    return 1;

}
