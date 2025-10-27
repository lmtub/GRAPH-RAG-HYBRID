static inline void gen_405_mulladd_insn(DisasContext *ctx, int opc2, int opc3,

                                        int ra, int rb, int rt, int Rc)

{

    TCGv t0, t1;



    t0 = tcg_temp_local_new();

    t1 = tcg_temp_local_new();



    switch (opc3 & 0x0D) {

    case 0x05:

        /* macchw    - macchw.    - macchwo   - macchwo.   */

        /* macchws   - macchws.   - macchwso  - macchwso.  */

        /* nmacchw   - nmacchw.   - nmacchwo  - nmacchwo.  */

        /* nmacchws  - nmacchws.  - nmacchwso - nmacchwso. */

        /* mulchw - mulchw. */

        tcg_gen_ext16s_tl(t0, cpu_gpr[ra]);

        tcg_gen_sari_tl(t1, cpu_gpr[rb], 16);

        tcg_gen_ext16s_tl(t1, t1);

        break;

    case 0x04:

        /* macchwu   - macchwu.   - macchwuo  - macchwuo.  */

        /* macchwsu  - macchwsu.  - macchwsuo - macchwsuo. */

        /* mulchwu - mulchwu. */

        tcg_gen_ext16u_tl(t0, cpu_gpr[ra]);

        tcg_gen_shri_tl(t1, cpu_gpr[rb], 16);

        tcg_gen_ext16u_tl(t1, t1);

        break;

    case 0x01:

        /* machhw    - machhw.    - machhwo   - machhwo.   */

        /* machhws   - machhws.   - machhwso  - machhwso.  */

        /* nmachhw   - nmachhw.   - nmachhwo  - nmachhwo.  */

        /* nmachhws  - nmachhws.  - nmachhwso - nmachhwso. */

        /* mulhhw - mulhhw. */

        tcg_gen_sari_tl(t0, cpu_gpr[ra], 16);

        tcg_gen_ext16s_tl(t0, t0);

        tcg_gen_sari_tl(t1, cpu_gpr[rb], 16);

        tcg_gen_ext16s_tl(t1, t1);

        break;

    case 0x00:

        /* machhwu   - machhwu.   - machhwuo  - machhwuo.  */

        /* machhwsu  - machhwsu.  - machhwsuo - machhwsuo. */

        /* mulhhwu - mulhhwu. */

        tcg_gen_shri_tl(t0, cpu_gpr[ra], 16);

        tcg_gen_ext16u_tl(t0, t0);

        tcg_gen_shri_tl(t1, cpu_gpr[rb], 16);

        tcg_gen_ext16u_tl(t1, t1);

        break;

    case 0x0D:

        /* maclhw    - maclhw.    - maclhwo   - maclhwo.   */

        /* maclhws   - maclhws.   - maclhwso  - maclhwso.  */

        /* nmaclhw   - nmaclhw.   - nmaclhwo  - nmaclhwo.  */

        /* nmaclhws  - nmaclhws.  - nmaclhwso - nmaclhwso. */

        /* mullhw - mullhw. */

        tcg_gen_ext16s_tl(t0, cpu_gpr[ra]);

        tcg_gen_ext16s_tl(t1, cpu_gpr[rb]);

        break;

    case 0x0C:

        /* maclhwu   - maclhwu.   - maclhwuo  - maclhwuo.  */

        /* maclhwsu  - maclhwsu.  - maclhwsuo - maclhwsuo. */

        /* mullhwu - mullhwu. */

        tcg_gen_ext16u_tl(t0, cpu_gpr[ra]);

        tcg_gen_ext16u_tl(t1, cpu_gpr[rb]);

        break;

    }

    if (opc2 & 0x04) {

        /* (n)multiply-and-accumulate (0x0C / 0x0E) */

        tcg_gen_mul_tl(t1, t0, t1);

        if (opc2 & 0x02) {

            /* nmultiply-and-accumulate (0x0E) */

            tcg_gen_sub_tl(t0, cpu_gpr[rt], t1);

        } else {

            /* multiply-and-accumulate (0x0C) */

            tcg_gen_add_tl(t0, cpu_gpr[rt], t1);

        }



        if (opc3 & 0x12) {

            /* Check overflow and/or saturate */

            int l1 = gen_new_label();



            if (opc3 & 0x10) {

                /* Start with XER OV disabled, the most likely case */

                tcg_gen_movi_tl(cpu_ov, 0);

            }

            if (opc3 & 0x01) {

                /* Signed */

                tcg_gen_xor_tl(t1, cpu_gpr[rt], t1);

                tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l1);

                tcg_gen_xor_tl(t1, cpu_gpr[rt], t0);

                tcg_gen_brcondi_tl(TCG_COND_LT, t1, 0, l1);

                if (opc3 & 0x02) {

                    /* Saturate */

                    tcg_gen_sari_tl(t0, cpu_gpr[rt], 31);

                    tcg_gen_xori_tl(t0, t0, 0x7fffffff);

                }

            } else {

                /* Unsigned */

                tcg_gen_brcond_tl(TCG_COND_GEU, t0, t1, l1);

                if (opc3 & 0x02) {

                    /* Saturate */

                    tcg_gen_movi_tl(t0, UINT32_MAX);

                }

            }

            if (opc3 & 0x10) {

                /* Check overflow */

                tcg_gen_movi_tl(cpu_ov, 1);

                tcg_gen_movi_tl(cpu_so, 1);

            }

            gen_set_label(l1);

            tcg_gen_mov_tl(cpu_gpr[rt], t0);

        }

    } else {

        tcg_gen_mul_tl(cpu_gpr[rt], t0, t1);

    }

    tcg_temp_free(t0);

    tcg_temp_free(t1);

    if (unlikely(Rc) != 0) {

        /* Update Rc0 */

        gen_set_Rc0(ctx, cpu_gpr[rt]);

    }

}
