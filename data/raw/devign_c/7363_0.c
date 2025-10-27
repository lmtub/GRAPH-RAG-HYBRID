static void gen_mtc0(DisasContext *ctx, TCGv arg, int reg, int sel)

{

    const char *rn = "invalid";



    if (sel != 0)

        check_insn(ctx, ISA_MIPS32);



    if (use_icount)

        gen_io_start();



    switch (reg) {

    case 0:

        switch (sel) {

        case 0:

            gen_helper_mtc0_index(cpu_env, arg);

            rn = "Index";

            break;

        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_mvpcontrol(cpu_env, arg);

            rn = "MVPControl";

            break;

        case 2:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            /* ignored */

            rn = "MVPConf0";

            break;

        case 3:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            /* ignored */

            rn = "MVPConf1";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 1:

        switch (sel) {

        case 0:

            /* ignored */

            rn = "Random";

            break;

        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_vpecontrol(cpu_env, arg);

            rn = "VPEControl";

            break;

        case 2:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_vpeconf0(cpu_env, arg);

            rn = "VPEConf0";

            break;

        case 3:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_vpeconf1(cpu_env, arg);

            rn = "VPEConf1";

            break;

        case 4:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_yqmask(cpu_env, arg);

            rn = "YQMask";

            break;

        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mtc0_store64(arg, offsetof(CPUMIPSState, CP0_VPESchedule));

            rn = "VPESchedule";

            break;

        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mtc0_store64(arg, offsetof(CPUMIPSState, CP0_VPEScheFBack));

            rn = "VPEScheFBack";

            break;

        case 7:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_vpeopt(cpu_env, arg);

            rn = "VPEOpt";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 2:

        switch (sel) {

        case 0:

            gen_helper_mtc0_entrylo0(cpu_env, arg);

            rn = "EntryLo0";

            break;

        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_tcstatus(cpu_env, arg);

            rn = "TCStatus";

            break;

        case 2:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_tcbind(cpu_env, arg);

            rn = "TCBind";

            break;

        case 3:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_tcrestart(cpu_env, arg);

            rn = "TCRestart";

            break;

        case 4:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_tchalt(cpu_env, arg);

            rn = "TCHalt";

            break;

        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_tccontext(cpu_env, arg);

            rn = "TCContext";

            break;

        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_tcschedule(cpu_env, arg);

            rn = "TCSchedule";

            break;

        case 7:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mtc0_tcschefback(cpu_env, arg);

            rn = "TCScheFBack";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 3:

        switch (sel) {

        case 0:

            gen_helper_mtc0_entrylo1(cpu_env, arg);

            rn = "EntryLo1";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 4:

        switch (sel) {

        case 0:

            gen_helper_mtc0_context(cpu_env, arg);

            rn = "Context";

            break;

        case 1:

//            gen_helper_mtc0_contextconfig(cpu_env, arg); /* SmartMIPS ASE */

            rn = "ContextConfig";

            goto cp0_unimplemented;

//            break;

        case 2:

            CP0_CHECK(ctx->ulri);

            tcg_gen_st_tl(arg, cpu_env,

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

            gen_helper_mtc0_pagemask(cpu_env, arg);

            rn = "PageMask";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_pagegrain(cpu_env, arg);

            rn = "PageGrain";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 6:

        switch (sel) {

        case 0:

            gen_helper_mtc0_wired(cpu_env, arg);

            rn = "Wired";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_srsconf0(cpu_env, arg);

            rn = "SRSConf0";

            break;

        case 2:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_srsconf1(cpu_env, arg);

            rn = "SRSConf1";

            break;

        case 3:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_srsconf2(cpu_env, arg);

            rn = "SRSConf2";

            break;

        case 4:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_srsconf3(cpu_env, arg);

            rn = "SRSConf3";

            break;

        case 5:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_srsconf4(cpu_env, arg);

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

            gen_helper_mtc0_hwrena(cpu_env, arg);

            ctx->bstate = BS_STOP;

            rn = "HWREna";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 8:

        switch (sel) {

        case 0:

            /* ignored */

            rn = "BadVAddr";

            break;

        case 1:

            /* ignored */

            rn = "BadInstr";

            break;

        case 2:

            /* ignored */

            rn = "BadInstrP";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 9:

        switch (sel) {

        case 0:

            gen_helper_mtc0_count(cpu_env, arg);

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

            gen_helper_mtc0_entryhi(cpu_env, arg);

            rn = "EntryHi";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 11:

        switch (sel) {

        case 0:

            gen_helper_mtc0_compare(cpu_env, arg);

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

            save_cpu_state(ctx, 1);

            gen_helper_mtc0_status(cpu_env, arg);

            /* BS_STOP isn't good enough here, hflags may have changed. */

            gen_save_pc(ctx->pc + 4);

            ctx->bstate = BS_EXCP;

            rn = "Status";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_intctl(cpu_env, arg);

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            rn = "IntCtl";

            break;

        case 2:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_srsctl(cpu_env, arg);

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            rn = "SRSCtl";

            break;

        case 3:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mtc0_store32(arg, offsetof(CPUMIPSState, CP0_SRSMap));

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            rn = "SRSMap";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 13:

        switch (sel) {

        case 0:

            save_cpu_state(ctx, 1);

            gen_helper_mtc0_cause(cpu_env, arg);

            rn = "Cause";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 14:

        switch (sel) {

        case 0:

            gen_mtc0_store64(arg, offsetof(CPUMIPSState, CP0_EPC));

            rn = "EPC";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 15:

        switch (sel) {

        case 0:

            /* ignored */

            rn = "PRid";

            break;

        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_helper_mtc0_ebase(cpu_env, arg);

            rn = "EBase";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 16:

        switch (sel) {

        case 0:

            gen_helper_mtc0_config0(cpu_env, arg);

            rn = "Config";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            break;

        case 1:

            /* ignored, read only */

            rn = "Config1";

            break;

        case 2:

            gen_helper_mtc0_config2(cpu_env, arg);

            rn = "Config2";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            break;

        case 3:

            gen_helper_mtc0_config3(cpu_env, arg);

            rn = "Config3";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            break;

        case 4:

            gen_helper_mtc0_config4(cpu_env, arg);

            rn = "Config4";

            ctx->bstate = BS_STOP;

            break;

        case 5:

            gen_helper_mtc0_config5(cpu_env, arg);

            rn = "Config5";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            break;

        /* 6,7 are implementation dependent */

        case 6:

            /* ignored */

            rn = "Config6";

            break;

        case 7:

            /* ignored */

            rn = "Config7";

            break;

        default:

            rn = "Invalid config selector";

            goto cp0_unimplemented;

        }

        break;

    case 17:

        switch (sel) {

        case 0:

            gen_helper_mtc0_lladdr(cpu_env, arg);

            rn = "LLAddr";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 18:

        switch (sel) {

        case 0 ... 7:

            gen_helper_0e1i(mtc0_watchlo, arg, sel);

            rn = "WatchLo";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 19:

        switch (sel) {

        case 0 ... 7:

            gen_helper_0e1i(mtc0_watchhi, arg, sel);

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

            gen_helper_mtc0_xcontext(cpu_env, arg);

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

            gen_helper_mtc0_framemask(cpu_env, arg);

            rn = "Framemask";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 22:

        /* ignored */

        rn = "Diagnostic"; /* implementation dependent */

        break;

    case 23:

        switch (sel) {

        case 0:

            gen_helper_mtc0_debug(cpu_env, arg); /* EJTAG support */

            /* BS_STOP isn't good enough here, hflags may have changed. */

            gen_save_pc(ctx->pc + 4);

            ctx->bstate = BS_EXCP;

            rn = "Debug";

            break;

        case 1:

//            gen_helper_mtc0_tracecontrol(cpu_env, arg); /* PDtrace support */

            rn = "TraceControl";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            break;

        case 2:

//            gen_helper_mtc0_tracecontrol2(cpu_env, arg); /* PDtrace support */

            rn = "TraceControl2";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            break;

        case 3:

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            gen_helper_mtc0_usertracedata(cpu_env, arg); /* PDtrace support */

            rn = "UserTraceData";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            break;

        case 4:

//            gen_helper_mtc0_tracebpc(cpu_env, arg); /* PDtrace support */

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

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

            gen_mtc0_store64(arg, offsetof(CPUMIPSState, CP0_DEPC));

            rn = "DEPC";

            break;

        default:

            goto cp0_unimplemented;

        }

        break;

    case 25:

        switch (sel) {

        case 0:

            gen_helper_mtc0_performance0(cpu_env, arg);

            rn = "Performance0";

            break;

        case 1:

//            gen_helper_mtc0_performance1(arg);

            rn = "Performance1";

//            break;

        case 2:

//            gen_helper_mtc0_performance2(arg);

            rn = "Performance2";

//            break;

        case 3:

//            gen_helper_mtc0_performance3(arg);

            rn = "Performance3";

//            break;

        case 4:

//            gen_helper_mtc0_performance4(arg);

            rn = "Performance4";

//            break;

        case 5:

//            gen_helper_mtc0_performance5(arg);

            rn = "Performance5";

//            break;

        case 6:

//            gen_helper_mtc0_performance6(arg);

            rn = "Performance6";

//            break;

        case 7:

//            gen_helper_mtc0_performance7(arg);

            rn = "Performance7";

//            break;

        default:

            goto cp0_unimplemented;

        }

       break;

    case 26:

        /* ignored */

        rn = "ECC";

        break;

    case 27:

        switch (sel) {

        case 0 ... 3:

            /* ignored */

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

            gen_helper_mtc0_taglo(cpu_env, arg);

            rn = "TagLo";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_helper_mtc0_datalo(cpu_env, arg);

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

            gen_helper_mtc0_taghi(cpu_env, arg);

            rn = "TagHi";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_helper_mtc0_datahi(cpu_env, arg);

            rn = "DataHi";

            break;

        default:

            rn = "invalid sel";

            goto cp0_unimplemented;

        }

       break;

    case 30:

        switch (sel) {

        case 0:

            gen_mtc0_store64(arg, offsetof(CPUMIPSState, CP0_ErrorEPC));

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

            gen_mtc0_store32(arg, offsetof(CPUMIPSState, CP0_DESAVE));

            rn = "DESAVE";

            break;

        case 2 ... 7:

            CP0_CHECK(ctx->kscrexist & (1 << sel));

            tcg_gen_st_tl(arg, cpu_env,

                          offsetof(CPUMIPSState, CP0_KScratch[sel-2]));

            rn = "KScratch";

            break;

        default:

            goto cp0_unimplemented;

        }

        /* Stop translation as we may have switched the execution mode */

        ctx->bstate = BS_STOP;

        break;

    default:

       goto cp0_unimplemented;

    }

    (void)rn; /* avoid a compiler warning */

    LOG_DISAS("mtc0 %s (reg %d sel %d)\n", rn, reg, sel);

    /* For simplicity assume that all writes can cause interrupts.  */

    if (use_icount) {

        gen_io_end();

        ctx->bstate = BS_STOP;

    }

    return;



cp0_unimplemented:

    LOG_DISAS("mtc0 %s (reg %d sel %d)\n", rn, reg, sel);

}
