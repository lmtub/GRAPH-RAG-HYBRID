static void gen_dmfc0(DisasContext *ctx, TCGv arg, int reg, int sel)

{

    const char *rn = "invalid";



    if (sel != 0)

        check_insn(ctx, ISA_MIPS64);



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

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_YQMask));

            rn = "YQMask";

            break;

        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_VPESchedule));

            rn = "VPESchedule";

            break;

        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_VPEScheFBack));

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

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EntryLo0));

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

            gen_helper_dmfc0_tcrestart(arg, cpu_env);

            rn = "TCRestart";

            break;

        case 4:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tchalt(arg, cpu_env);

            rn = "TCHalt";

            break;

        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tccontext(arg, cpu_env);

            rn = "TCContext";

            break;

        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tcschedule(arg, cpu_env);

            rn = "TCSchedule";

            break;

        case 7:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tcschefback(arg, cpu_env);

            rn = "TCScheFBack";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 3:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EntryLo1));

            rn = "EntryLo1";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 4:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_Context));

            rn = "Context";

            break;

        case 1:

//            gen_helper_dmfc0_contextconfig(arg); /* SmartMIPS ASE */

            rn = "ContextConfig";

            goto cp0_unimplemented;

//            break;

        case 2:

            CP0_CHECK(ctx->ulri);

            tcg_gen_ld_tl(arg, cpu_env,

                          offsetof(CPUMIPSState, active_tc.CP0_UserLocal));

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

            if (use_icount)

                gen_io_start();

            gen_helper_mfc0_count(arg, cpu_env);

            if (use_icount) {

                gen_io_end();

            }

            /* Break the TB to be able to take timer interrupts immediately

               after reading count.  */

            ctx->bstate = BS_STOP;

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

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_EBase));

            rn = "EBase";

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

            gen_helper_dmfc0_lladdr(arg, cpu_env);

            rn = "LLAddr";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 18:

        switch (sel) {

        case 0 ... 7:

            gen_helper_1e0i(dmfc0_watchlo, arg, sel);

            rn = "WatchLo";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 19:

        switch (sel) {

        case 0 ... 7:

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

            check_insn(ctx, ISA_MIPS3);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_XContext));

            rn = "XContext";

            break;

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

//            gen_helper_dmfc0_tracecontrol(arg, cpu_env); /* PDtrace support */

            rn = "TraceControl";

//            break;

        case 2:

//            gen_helper_dmfc0_tracecontrol2(arg, cpu_env); /* PDtrace support */

            rn = "TraceControl2";

//            break;

        case 3:

//            gen_helper_dmfc0_usertracedata(arg, cpu_env); /* PDtrace support */

            rn = "UserTraceData";

//            break;

        case 4:

//            gen_helper_dmfc0_tracebpc(arg, cpu_env); /* PDtrace support */

            rn = "TraceBPC";

//            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 24:

        switch (sel) {

        case 0:

            /* EJTAG support */

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_DEPC));

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

//            gen_helper_dmfc0_performance1(arg);

            rn = "Performance1";

//            break;

        case 2:

//            gen_helper_dmfc0_performance2(arg);

            rn = "Performance2";

//            break;

        case 3:

//            gen_helper_dmfc0_performance3(arg);

            rn = "Performance3";

//            break;

        case 4:

//            gen_helper_dmfc0_performance4(arg);

            rn = "Performance4";

//            break;

        case 5:

//            gen_helper_dmfc0_performance5(arg);

            rn = "Performance5";

//            break;

        case 6:

//            gen_helper_dmfc0_performance6(arg);

            rn = "Performance6";

//            break;

        case 7:

//            gen_helper_dmfc0_performance7(arg);

            rn = "Performance7";

//            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 26:

        tcg_gen_movi_tl(arg, 0); /* unimplemented */

        rn = "ECC";

        break;

    case 27:

        switch (sel) {

        /* ignored */

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

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_TagLo));

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

            rn = "KScratch";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    default:

        goto cp0_unimplemented;

    }

    (void)rn; /* avoid a compiler warning */

    LOG_DISAS("dmfc0 %s (reg %d sel %d)\n", rn, reg, sel);

    return;



cp0_unimplemented:

    LOG_DISAS("dmfc0 %s (reg %d sel %d)\n", rn, reg, sel);

    gen_mfc0_unimplemented(ctx, arg);

}
