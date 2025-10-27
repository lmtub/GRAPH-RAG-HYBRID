static void gen_mftr(CPUMIPSState *env, DisasContext *ctx, int rt, int rd,

                     int u, int sel, int h)

{

    int other_tc = env->CP0_VPEControl & (0xff << CP0VPECo_TargTC);

    TCGv t0 = tcg_temp_local_new();



    if ((env->CP0_VPEConf0 & (1 << CP0VPEC0_MVP)) == 0 &&

        ((env->tcs[other_tc].CP0_TCBind & (0xf << CP0TCBd_CurVPE)) !=

         (env->active_tc.CP0_TCBind & (0xf << CP0TCBd_CurVPE))))

        tcg_gen_movi_tl(t0, -1);

    else if ((env->CP0_VPEControl & (0xff << CP0VPECo_TargTC)) >

             (env->mvp->CP0_MVPConf0 & (0xff << CP0MVPC0_PTC)))

        tcg_gen_movi_tl(t0, -1);

    else if (u == 0) {

        switch (rt) {

        case 1:

            switch (sel) {

            case 1:

                gen_helper_mftc0_vpecontrol(t0, cpu_env);

                break;

            case 2:

                gen_helper_mftc0_vpeconf0(t0, cpu_env);

                break;

            default:

                goto die;

                break;

            }

            break;

        case 2:

            switch (sel) {

            case 1:

                gen_helper_mftc0_tcstatus(t0, cpu_env);

                break;

            case 2:

                gen_helper_mftc0_tcbind(t0, cpu_env);

                break;

            case 3:

                gen_helper_mftc0_tcrestart(t0, cpu_env);

                break;

            case 4:

                gen_helper_mftc0_tchalt(t0, cpu_env);

                break;

            case 5:

                gen_helper_mftc0_tccontext(t0, cpu_env);

                break;

            case 6:

                gen_helper_mftc0_tcschedule(t0, cpu_env);

                break;

            case 7:

                gen_helper_mftc0_tcschefback(t0, cpu_env);

                break;

            default:

                gen_mfc0(ctx, t0, rt, sel);

                break;

            }

            break;

        case 10:

            switch (sel) {

            case 0:

                gen_helper_mftc0_entryhi(t0, cpu_env);

                break;

            default:

                gen_mfc0(ctx, t0, rt, sel);

                break;

            }

        case 12:

            switch (sel) {

            case 0:

                gen_helper_mftc0_status(t0, cpu_env);

                break;

            default:

                gen_mfc0(ctx, t0, rt, sel);

                break;

            }

        case 13:

            switch (sel) {

            case 0:

                gen_helper_mftc0_cause(t0, cpu_env);

                break;

            default:

                goto die;

                break;

            }

            break;

        case 14:

            switch (sel) {

            case 0:

                gen_helper_mftc0_epc(t0, cpu_env);

                break;

            default:

                goto die;

                break;

            }

            break;

        case 15:

            switch (sel) {

            case 1:

                gen_helper_mftc0_ebase(t0, cpu_env);

                break;

            default:

                goto die;

                break;

            }

            break;

        case 16:

            switch (sel) {

            case 0 ... 7:

                gen_helper_mftc0_configx(t0, cpu_env, tcg_const_tl(sel));

                break;

            default:

                goto die;

                break;

            }

            break;

        case 23:

            switch (sel) {

            case 0:

                gen_helper_mftc0_debug(t0, cpu_env);

                break;

            default:

                gen_mfc0(ctx, t0, rt, sel);

                break;

            }

            break;

        default:

            gen_mfc0(ctx, t0, rt, sel);

        }

    } else switch (sel) {

    /* GPR registers. */

    case 0:

        gen_helper_1e0i(mftgpr, t0, rt);

        break;

    /* Auxiliary CPU registers */

    case 1:

        switch (rt) {

        case 0:

            gen_helper_1e0i(mftlo, t0, 0);

            break;

        case 1:

            gen_helper_1e0i(mfthi, t0, 0);

            break;

        case 2:

            gen_helper_1e0i(mftacx, t0, 0);

            break;

        case 4:

            gen_helper_1e0i(mftlo, t0, 1);

            break;

        case 5:

            gen_helper_1e0i(mfthi, t0, 1);

            break;

        case 6:

            gen_helper_1e0i(mftacx, t0, 1);

            break;

        case 8:

            gen_helper_1e0i(mftlo, t0, 2);

            break;

        case 9:

            gen_helper_1e0i(mfthi, t0, 2);

            break;

        case 10:

            gen_helper_1e0i(mftacx, t0, 2);

            break;

        case 12:

            gen_helper_1e0i(mftlo, t0, 3);

            break;

        case 13:

            gen_helper_1e0i(mfthi, t0, 3);

            break;

        case 14:

            gen_helper_1e0i(mftacx, t0, 3);

            break;

        case 16:

            gen_helper_mftdsp(t0, cpu_env);

            break;

        default:

            goto die;

        }

        break;

    /* Floating point (COP1). */

    case 2:

        /* XXX: For now we support only a single FPU context. */

        if (h == 0) {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, rt);

            tcg_gen_ext_i32_tl(t0, fp0);

            tcg_temp_free_i32(fp0);

        } else {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32h(fp0, rt);

            tcg_gen_ext_i32_tl(t0, fp0);

            tcg_temp_free_i32(fp0);

        }

        break;

    case 3:

        /* XXX: For now we support only a single FPU context. */

        gen_helper_1e0i(cfc1, t0, rt);

        break;

    /* COP2: Not implemented. */

    case 4:

    case 5:

        /* fall through */

    default:

        goto die;

    }

    LOG_DISAS("mftr (reg %d u %d sel %d h %d)\n", rt, u, sel, h);

    gen_store_gpr(t0, rd);

    tcg_temp_free(t0);

    return;



die:

    tcg_temp_free(t0);

    LOG_DISAS("mftr (reg %d u %d sel %d h %d)\n", rt, u, sel, h);

    generate_exception(ctx, EXCP_RI);

}
