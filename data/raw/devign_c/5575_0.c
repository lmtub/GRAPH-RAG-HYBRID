void arm_gen_test_cc(int cc, int label)

{

    TCGv_i32 tmp;

    int inv;



    switch (cc) {

    case 0: /* eq: Z */

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_ZF, 0, label);

        break;

    case 1: /* ne: !Z */

        tcg_gen_brcondi_i32(TCG_COND_NE, cpu_ZF, 0, label);

        break;

    case 2: /* cs: C */

        tcg_gen_brcondi_i32(TCG_COND_NE, cpu_CF, 0, label);

        break;

    case 3: /* cc: !C */

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_CF, 0, label);

        break;

    case 4: /* mi: N */

        tcg_gen_brcondi_i32(TCG_COND_LT, cpu_NF, 0, label);

        break;

    case 5: /* pl: !N */

        tcg_gen_brcondi_i32(TCG_COND_GE, cpu_NF, 0, label);

        break;

    case 6: /* vs: V */

        tcg_gen_brcondi_i32(TCG_COND_LT, cpu_VF, 0, label);

        break;

    case 7: /* vc: !V */

        tcg_gen_brcondi_i32(TCG_COND_GE, cpu_VF, 0, label);

        break;

    case 8: /* hi: C && !Z */

        inv = gen_new_label();

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_CF, 0, inv);

        tcg_gen_brcondi_i32(TCG_COND_NE, cpu_ZF, 0, label);

        gen_set_label(inv);

        break;

    case 9: /* ls: !C || Z */

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_CF, 0, label);

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_ZF, 0, label);

        break;

    case 10: /* ge: N == V -> N ^ V == 0 */

        tmp = tcg_temp_new_i32();

        tcg_gen_xor_i32(tmp, cpu_VF, cpu_NF);

        tcg_gen_brcondi_i32(TCG_COND_GE, tmp, 0, label);

        tcg_temp_free_i32(tmp);

        break;

    case 11: /* lt: N != V -> N ^ V != 0 */

        tmp = tcg_temp_new_i32();

        tcg_gen_xor_i32(tmp, cpu_VF, cpu_NF);

        tcg_gen_brcondi_i32(TCG_COND_LT, tmp, 0, label);

        tcg_temp_free_i32(tmp);

        break;

    case 12: /* gt: !Z && N == V */

        inv = gen_new_label();

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_ZF, 0, inv);

        tmp = tcg_temp_new_i32();

        tcg_gen_xor_i32(tmp, cpu_VF, cpu_NF);

        tcg_gen_brcondi_i32(TCG_COND_GE, tmp, 0, label);

        tcg_temp_free_i32(tmp);

        gen_set_label(inv);

        break;

    case 13: /* le: Z || N != V */

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_ZF, 0, label);

        tmp = tcg_temp_new_i32();

        tcg_gen_xor_i32(tmp, cpu_VF, cpu_NF);

        tcg_gen_brcondi_i32(TCG_COND_LT, tmp, 0, label);

        tcg_temp_free_i32(tmp);

        break;

    default:

        fprintf(stderr, "Bad condition code 0x%x\n", cc);

        abort();

    }

}
