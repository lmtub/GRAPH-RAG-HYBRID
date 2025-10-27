static void gen_mfc0(DisasContext *ctx, TCGv arg, int reg, int sel)

{

    const char *rn = "invalid";



    if (sel != 0)

        check_insn(ctx, ISA_MIPS32);



    switch (reg) {

    case 0:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Index));

            rn = "Index";

            break;

        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_mvpcontrol(arg, cpu_env);

            rn = "MVPControl";

            break;

        case 2:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_mvpconf0(arg, cpu_env);

            rn = "MVPConf0";

            break;

        case 3:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_mvpconf1(arg, cpu_env);

            rn = "MVPConf1";

            break;

        case 4:

            CP0_CHECK(ctx->vp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPControl));

            rn = "VPControl";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 1:

        switch (sel) {

        case 0:

            CP0_CHECK(!(ctx->insn_flags & ISA_MIPS32R6));

            gen_helper_mfc0_random(arg, cpu_env);

            rn = "Random";

            break;

        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEControl));

            rn = "VPEControl";

            break;

        case 2:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEConf0));

            rn = "VPEConf0";

            break;

        case 3:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEConf1));

            rn = "VPEConf1";

            break;

        case 4:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load64(arg, offsetof(CPUMIPSState, CP0_YQMask));

            rn = "YQMask";

            break;

        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load64(arg, offsetof(CPUMIPSState, CP0_VPESchedule));

            rn = "VPESchedule";

            break;

        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load64(arg, offsetof(CPUMIPSState, CP0_VPEScheFBack));

            rn = "VPEScheFBack";

            break;

        case 7:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEOpt));

            rn = "VPEOpt";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 2:

        switch (sel) {

        case 0:

            {

                TCGv_i64 tmp = tcg_temp_new_i64();

                tcg_gen_ld_i64(tmp, cpu_env,

                               offsetof(CPUMIPSState, CP0_EntryLo0));

#if defined(TARGET_MIPS64)

                if (ctx->rxi) {

                    /* Move RI/XI fields to bits 31:30 */

                    tcg_gen_shri_tl(arg, tmp, CP0EnLo_XI);

                    tcg_gen_deposit_tl(tmp, tmp, arg, 30, 2);

                }

#endif

                gen_move_low32(arg, tmp);

                tcg_temp_free_i64(tmp);

            }

            rn = "EntryLo0";

            break;

        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tcstatus(arg, cpu_env);

            rn = "TCStatus";

            break;

        case 2:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tcbind(arg, cpu_env);

            rn = "TCBind";

            break;

        case 3:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tcrestart(arg, cpu_env);

            rn = "TCRestart";

            break;

        case 4:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tchalt(arg, cpu_env);

            rn = "TCHalt";

            break;

        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tccontext(arg, cpu_env);

            rn = "TCContext";

            break;

        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tcschedule(arg, cpu_env);

            rn = "TCSchedule";

            break;

        case 7:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tcschefback(arg, cpu_env);

            rn = "TCScheFBack";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 3:

        switch (sel) {

        case 0:

            {

                TCGv_i64 tmp = tcg_temp_new_i64();

                tcg_gen_ld_i64(tmp, cpu_env,

                               offsetof(CPUMIPSState, CP0_EntryLo1));

#if defined(TARGET_MIPS64)

                if (ctx->rxi) {

                    /* Move RI/XI fields to bits 31:30 */

                    tcg_gen_shri_tl(arg, tmp, CP0EnLo_XI);

                    tcg_gen_deposit_tl(tmp, tmp, arg, 30, 2);

                }

#endif

                gen_move_low32(arg, tmp);

                tcg_temp_free_i64(tmp);

            }

            rn = "EntryLo1";

            break;

        case 1:

            CP0_CHECK(ctx->vp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_GlobalNumber));

            rn = "GlobalNumber";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 4:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_Context));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "Context";

            break;

        case 1:

//            gen_helper_mfc0_contextconfig(arg); /* SmartMIPS ASE */

            rn = "ContextConfig";

            goto cp0_unimplemented;

        case 2:

            CP0_CHECK(ctx->ulri);

            tcg_gen_ld_tl(arg, cpu_env,

                          offsetof(CPUMIPSState, active_tc.CP0_UserLocal));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "UserLocal";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 5:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_PageMask));

            rn = "PageMask";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_PageGrain));

            rn = "PageGrain";

            break;

        case 2:

            CP0_CHECK(ctx->sc);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_SegCtl0));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "SegCtl0";

            break;

        case 3:

            CP0_CHECK(ctx->sc);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_SegCtl1));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "SegCtl1";

            break;

        case 4:

            CP0_CHECK(ctx->sc);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_SegCtl2));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "SegCtl2";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 6:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Wired));

            rn = "Wired";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf0));

            rn = "SRSConf0";

            break;

        case 2:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf1));

            rn = "SRSConf1";

            break;

        case 3:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf2));

            rn = "SRSConf2";

            break;

        case 4:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf3));

            rn = "SRSConf3";

            break;

        case 5:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf4));

            rn = "SRSConf4";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 7:

        switch (sel) {

        case 0:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_HWREna));

            rn = "HWREna";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 8:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_BadVAddr));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "BadVAddr";

            break;

        case 1:

            CP0_CHECK(ctx->bi);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_BadInstr));

            rn = "BadInstr";

            break;

        case 2:

            CP0_CHECK(ctx->bp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_BadInstrP));

            rn = "BadInstrP";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 9:

        switch (sel) {

        case 0:

            /* Mark as an IO operation because we read the time.  */

            if (ctx->tb->cflags & CF_USE_ICOUNT) {

                gen_io_start();

	    }

            gen_helper_mfc0_count(arg, cpu_env);

            if (ctx->tb->cflags & CF_USE_ICOUNT) {

                gen_io_end();

            }

            /* Break the TB to be able to take timer interrupts immediately

               after reading count. BS_STOP isn't sufficient, we need to ensure

               we break completely out of translated code.  */

            gen_save_pc(ctx->pc + 4);

            ctx->bstate = BS_EXCP;

            rn = "Count";

            break;

        /* 6,7 are implementation dependent */

        default:

            goto cp0_unimplemented;

        }

        break;

    case 10:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EntryHi));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "EntryHi";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 11:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Compare));

            rn = "Compare";

            break;

        /* 6,7 are implementation dependent */

        default:

            goto cp0_unimplemented;

        }

        break;

    case 12:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Status));

            rn = "Status";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_IntCtl));

            rn = "IntCtl";

            break;

        case 2:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSCtl));

            rn = "SRSCtl";

            break;

        case 3:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSMap));

            rn = "SRSMap";

            break;

        default:

            goto cp0_unimplemented;

       }

        break;

    case 13:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Cause));

            rn = "Cause";

            break;

        default:

            goto cp0_unimplemented;

       }

        break;

    case 14:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EPC));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "EPC";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 15:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_PRid));

            rn = "PRid";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EBase));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "EBase";

            break;

        case 3:

            check_insn(ctx, ISA_MIPS32R2);

            CP0_CHECK(ctx->cmgcr);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_CMGCRBase));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "CMGCRBase";

            break;

        default:

            goto cp0_unimplemented;

       }

        break;

    case 16:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config0));

            rn = "Config";

            break;

        case 1:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config1));

            rn = "Config1";

            break;

        case 2:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config2));

            rn = "Config2";

            break;

        case 3:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config3));

            rn = "Config3";

            break;

        case 4:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config4));

            rn = "Config4";

            break;

        case 5:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config5));

            rn = "Config5";

            break;

        /* 6,7 are implementation dependent */

        case 6:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config6));

            rn = "Config6";

            break;

        case 7:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config7));

            rn = "Config7";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 17:

        switch (sel) {

        case 0:

            gen_helper_mfc0_lladdr(arg, cpu_env);

            rn = "LLAddr";

            break;

        case 1:

            CP0_CHECK(ctx->mrp);

            gen_helper_mfc0_maar(arg, cpu_env);

            rn = "MAAR";

            break;

        case 2:

            CP0_CHECK(ctx->mrp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_MAARI));

            rn = "MAARI";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 18:

        switch (sel) {

        case 0 ... 7:

            gen_helper_1e0i(mfc0_watchlo, arg, sel);

            rn = "WatchLo";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 19:

        switch (sel) {

        case 0 ...7:

            gen_helper_1e0i(mfc0_watchhi, arg, sel);

            rn = "WatchHi";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 20:

        switch (sel) {

        case 0:

#if defined(TARGET_MIPS64)

            check_insn(ctx, ISA_MIPS3);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_XContext));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "XContext";

            break;

#endif

        default:

            goto cp0_unimplemented;

        }

        break;

    case 21:

       /* Officially reserved, but sel 0 is used for R1x000 framemask */

        CP0_CHECK(!(ctx->insn_flags & ISA_MIPS32R6));

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Framemask));

            rn = "Framemask";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 22:

        tcg_gen_movi_tl(arg, 0); /* unimplemented */

        rn = "'Diagnostic"; /* implementation dependent */

        break;

    case 23:

        switch (sel) {

        case 0:

            gen_helper_mfc0_debug(arg, cpu_env); /* EJTAG support */

            rn = "Debug";

            break;

        case 1:

//            gen_helper_mfc0_tracecontrol(arg); /* PDtrace support */

            rn = "TraceControl";

            goto cp0_unimplemented;

        case 2:

//            gen_helper_mfc0_tracecontrol2(arg); /* PDtrace support */

            rn = "TraceControl2";

            goto cp0_unimplemented;

        case 3:

//            gen_helper_mfc0_usertracedata(arg); /* PDtrace support */

            rn = "UserTraceData";

            goto cp0_unimplemented;

        case 4:

//            gen_helper_mfc0_tracebpc(arg); /* PDtrace support */

            rn = "TraceBPC";

            goto cp0_unimplemented;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 24:

        switch (sel) {

        case 0:

            /* EJTAG support */

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_DEPC));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "DEPC";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 25:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Performance0));

            rn = "Performance0";

            break;

        case 1:

//            gen_helper_mfc0_performance1(arg);

            rn = "Performance1";

            goto cp0_unimplemented;

        case 2:

//            gen_helper_mfc0_performance2(arg);

            rn = "Performance2";

            goto cp0_unimplemented;

        case 3:

//            gen_helper_mfc0_performance3(arg);

            rn = "Performance3";

            goto cp0_unimplemented;

        case 4:

//            gen_helper_mfc0_performance4(arg);

            rn = "Performance4";

            goto cp0_unimplemented;

        case 5:

//            gen_helper_mfc0_performance5(arg);

            rn = "Performance5";

            goto cp0_unimplemented;

        case 6:

//            gen_helper_mfc0_performance6(arg);

            rn = "Performance6";

            goto cp0_unimplemented;

        case 7:

//            gen_helper_mfc0_performance7(arg);

            rn = "Performance7";

            goto cp0_unimplemented;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 26:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_ErrCtl));

            rn = "ErrCtl";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 27:

        switch (sel) {

        case 0 ... 3:

            tcg_gen_movi_tl(arg, 0); /* unimplemented */

            rn = "CacheErr";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 28:

        switch (sel) {

        case 0:

        case 2:

        case 4:

        case 6:

            {

                TCGv_i64 tmp = tcg_temp_new_i64();

                tcg_gen_ld_i64(tmp, cpu_env, offsetof(CPUMIPSState, CP0_TagLo));

                gen_move_low32(arg, tmp);

                tcg_temp_free_i64(tmp);

            }

            rn = "TagLo";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_DataLo));

            rn = "DataLo";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 29:

        switch (sel) {

        case 0:

        case 2:

        case 4:

        case 6:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_TagHi));

            rn = "TagHi";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_DataHi));

            rn = "DataHi";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 30:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_ErrorEPC));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "ErrorEPC";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 31:

        switch (sel) {

        case 0:

            /* EJTAG support */

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_DESAVE));

            rn = "DESAVE";

            break;

        case 2 ... 7:

            CP0_CHECK(ctx->kscrexist & (1 << sel));

            tcg_gen_ld_tl(arg, cpu_env,

                          offsetof(CPUMIPSState, CP0_KScratch[sel-2]));

            tcg_gen_ext32s_tl(arg, arg);

            rn = "KScratch";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    default:

       goto cp0_unimplemented;

    }

    trace_mips_translate_c0("mfc0", rn, reg, sel);

    return;



cp0_unimplemented:

    qemu_log_mask(LOG_UNIMP, "mfc0 %s (reg %d sel %d)\n", rn, reg, sel);

    gen_mfc0_unimplemented(ctx, arg);

}
