static void gen_mttr(CPUMIPSState *env, DisasContext *ctx, int rd, int rt,

                     int u, int sel, int h)

{

    int other_tc = env->CP0_VPEControl & (0xff << CP0VPECo_TargTC);

    TCGv t0 = tcg_temp_local_new();



    gen_load_gpr(t0, rt);

    if ((env->CP0_VPEConf0 & (1 << CP0VPEC0_MVP)) == 0 &&

        ((env->tcs[other_tc].CP0_TCBind & (0xf << CP0TCBd_CurVPE)) !=

         (env->active_tc.CP0_TCBind & (0xf << CP0TCBd_CurVPE))))

        /* NOP */ ;

    else if ((env->CP0_VPEControl & (0xff << CP0VPECo_TargTC)) >

             (env->mvp->CP0_MVPConf0 & (0xff << CP0MVPC0_PTC)))

        /* NOP */ ;

    else if (u == 0) {

        switch (rd) {

        case 1:

            switch (sel) {

            case 1:

                gen_helper_mttc0_vpecontrol(cpu_env, t0);

                break;

            case 2:

                gen_helper_mttc0_vpeconf0(cpu_env, t0);

                break;

            default:

                goto die;

                break;

            }

            break;

        case 2:

            switch (sel) {

            case 1:

                gen_helper_mttc0_tcstatus(cpu_env, t0);

                break;

            case 2:

                gen_helper_mttc0_tcbind(cpu_env, t0);

                break;

            case 3:

                gen_helper_mttc0_tcrestart(cpu_env, t0);

                break;

            case 4:

                gen_helper_mttc0_tchalt(cpu_env, t0);

                break;

            case 5:

                gen_helper_mttc0_tccontext(cpu_env, t0);

                break;

            case 6:

                gen_helper_mttc0_tcschedule(cpu_env, t0);

                break;

            case 7:

                gen_helper_mttc0_tcschefback(cpu_env, t0);

                break;

            default:

                gen_mtc0(ctx, t0, rd, sel);

                break;

            }

            break;

        case 10:

            switch (sel) {

            case 0:

                gen_helper_mttc0_entryhi(cpu_env, t0);

                break;

            default:

                gen_mtc0(ctx, t0, rd, sel);

                break;

            }

        case 12:

            switch (sel) {

            case 0:

                gen_helper_mttc0_status(cpu_env, t0);

                break;

            default:

                gen_mtc0(ctx, t0, rd, sel);

                break;

            }

        case 13:

            switch (sel) {

            case 0:

                gen_helper_mttc0_cause(cpu_env, t0);

                break;

            default:

                goto die;

                break;

            }

            break;

        case 15:

            switch (sel) {

            case 1:

                gen_helper_mttc0_ebase(cpu_env, t0);

                break;

            default:

                goto die;

                break;

            }

            break;

        case 23:

            switch (sel) {

            case 0:

                gen_helper_mttc0_debug(cpu_env, t0);

                break;

            default:

                gen_mtc0(ctx, t0, rd, sel);

                break;

            }

            break;

        default:

            gen_mtc0(ctx, t0, rd, sel);

        }

    } else switch (sel) {

    /* GPR registers. */

    case 0:

        gen_helper_0e1i(mttgpr, t0, rd);

        break;

    /* Auxiliary CPU registers */

    case 1:

        switch (rd) {

        case 0:

            gen_helper_0e1i(mttlo, t0, 0);

            break;

        case 1:

            gen_helper_0e1i(mtthi, t0, 0);

            break;

        case 2:

            gen_helper_0e1i(mttacx, t0, 0);

            break;

        case 4:

            gen_helper_0e1i(mttlo, t0, 1);

            break;

        case 5:

            gen_helper_0e1i(mtthi, t0, 1);

            break;

        case 6:

            gen_helper_0e1i(mttacx, t0, 1);

            break;

        case 8:

            gen_helper_0e1i(mttlo, t0, 2);

            break;

        case 9:

            gen_helper_0e1i(mtthi, t0, 2);

            break;

        case 10:

            gen_helper_0e1i(mttacx, t0, 2);

            break;

        case 12:

            gen_helper_0e1i(mttlo, t0, 3);

            break;

        case 13:

            gen_helper_0e1i(mtthi, t0, 3);

            break;

        case 14:

            gen_helper_0e1i(mttacx, t0, 3);

            break;

        case 16:

            gen_helper_mttdsp(cpu_env, t0);

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



            tcg_gen_trunc_tl_i32(fp0, t0);

            gen_store_fpr32(fp0, rd);

            tcg_temp_free_i32(fp0);

        } else {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            tcg_gen_trunc_tl_i32(fp0, t0);

            gen_store_fpr32h(fp0, rd);

            tcg_temp_free_i32(fp0);

        }

        break;

    case 3:

        /* XXX: For now we support only a single FPU context. */

        {

            TCGv_i32 fs_tmp = tcg_const_i32(rd);



            gen_helper_0e2i(ctc1, t0, fs_tmp, rt);

            tcg_temp_free_i32(fs_tmp);

        }

        break;

    /* COP2: Not implemented. */

    case 4:

    case 5:

        /* fall through */

    default:

        goto die;

    }

    LOG_DISAS("mttr (reg %d u %d sel %d h %d)\n", rd, u, sel, h);

    tcg_temp_free(t0);

    return;



die:

    tcg_temp_free(t0);

    LOG_DISAS("mttr (reg %d u %d sel %d h %d)\n", rd, u, sel, h);

    generate_exception(ctx, EXCP_RI);

}
