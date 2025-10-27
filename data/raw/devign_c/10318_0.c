static void gen_mtc0 (DisasContext *ctx, int reg, int sel)

{

    const char *rn = "invalid";



    switch (reg) {

    case 0:

        switch (sel) {

        case 0:

           gen_op_mtc0_index();

            rn = "Index";

            break;

        case 1:

//            gen_op_mtc0_mvpcontrol(); /* MT ASE */

            rn = "MVPControl";

//            break;

        case 2:

//            gen_op_mtc0_mvpconf0(); /* MT ASE */

            rn = "MVPConf0";

//            break;

        case 3:

//            gen_op_mtc0_mvpconf1(); /* MT ASE */

            rn = "MVPConf1";

//            break;

        default:

            goto die;

        }

        break;

    case 1:

        switch (sel) {

        case 0:

            /* ignored */

            rn = "Random";

            break;

        case 1:

//            gen_op_mtc0_vpecontrol(); /* MT ASE */

            rn = "VPEControl";

//            break;

        case 2:

//            gen_op_mtc0_vpeconf0(); /* MT ASE */

            rn = "VPEConf0";

//            break;

        case 3:

//            gen_op_mtc0_vpeconf1(); /* MT ASE */

            rn = "VPEConf1";

//            break;

        case 4:

//            gen_op_mtc0_YQMask(); /* MT ASE */

            rn = "YQMask";

//            break;

        case 5:

//            gen_op_mtc0_vpeschedule(); /* MT ASE */

            rn = "VPESchedule";

//            break;

        case 6:

//            gen_op_mtc0_vpeschefback(); /* MT ASE */

            rn = "VPEScheFBack";

//            break;

        case 7:

//            gen_op_mtc0_vpeopt(); /* MT ASE */

            rn = "VPEOpt";

//            break;

        default:

            goto die;

        }

        break;

    case 2:

        switch (sel) {

        case 0:

            gen_op_mtc0_entrylo0();

            rn = "EntryLo0";

            break;

        case 1:

//            gen_op_mtc0_tcstatus(); /* MT ASE */

            rn = "TCStatus";

//            break;

        case 2:

//            gen_op_mtc0_tcbind(); /* MT ASE */

            rn = "TCBind";

//            break;

        case 3:

//            gen_op_mtc0_tcrestart(); /* MT ASE */

            rn = "TCRestart";

//            break;

        case 4:

//            gen_op_mtc0_tchalt(); /* MT ASE */

            rn = "TCHalt";

//            break;

        case 5:

//            gen_op_mtc0_tccontext(); /* MT ASE */

            rn = "TCContext";

//            break;

        case 6:

//            gen_op_mtc0_tcschedule(); /* MT ASE */

            rn = "TCSchedule";

//            break;

        case 7:

//            gen_op_mtc0_tcschefback(); /* MT ASE */

            rn = "TCScheFBack";

//            break;

        default:

            goto die;

        }

        break;

    case 3:

        switch (sel) {

        case 0:

            gen_op_mtc0_entrylo1();

            rn = "EntryLo1";

            break;

        default:

            goto die;

        }

        break;

    case 4:

        switch (sel) {

        case 0:

            gen_op_mtc0_context();

            rn = "Context";

            break;

        case 1:

//            gen_op_mtc0_contextconfig(); /* SmartMIPS ASE */

            rn = "ContextConfig";

//            break;

        default:

            goto die;

        }

        break;

    case 5:

        switch (sel) {

        case 0:

            gen_op_mtc0_pagemask();

            rn = "PageMask";

            break;

        case 1:

            gen_op_mtc0_pagegrain();

            rn = "PageGrain";

            break;

        default:

            goto die;

        }

        break;

    case 6:

        switch (sel) {

        case 0:

            gen_op_mtc0_wired();

            rn = "Wired";

            break;

        case 1:

//            gen_op_mtc0_srsconf0(); /* shadow registers */

            rn = "SRSConf0";

//            break;

        case 2:

//            gen_op_mtc0_srsconf1(); /* shadow registers */

            rn = "SRSConf1";

//            break;

        case 3:

//            gen_op_mtc0_srsconf2(); /* shadow registers */

            rn = "SRSConf2";

//            break;

        case 4:

//            gen_op_mtc0_srsconf3(); /* shadow registers */

            rn = "SRSConf3";

//            break;

        case 5:

//            gen_op_mtc0_srsconf4(); /* shadow registers */

            rn = "SRSConf4";

//            break;

        default:

            goto die;

        }

        break;

    case 7:

        switch (sel) {

        case 0:

            gen_op_mtc0_hwrena();

            rn = "HWREna";

            break;

        default:

            goto die;

        }

        break;

    case 8:

        /* ignored */

        rn = "BadVaddr";

        break;

    case 9:

        switch (sel) {

        case 0:

            gen_op_mtc0_count();

            rn = "Count";

            break;

        /* 6,7 are implementation dependent */

        default:

            goto die;

        }

        /* Stop translation as we may have switched the execution mode */

        ctx->bstate = BS_STOP;

        break;

    case 10:

        switch (sel) {

        case 0:

            gen_op_mtc0_entryhi();

            rn = "EntryHi";

            break;

        default:

            goto die;

        }

        break;

    case 11:

        switch (sel) {

        case 0:

            gen_op_mtc0_compare();

            rn = "Compare";

            break;

        /* 6,7 are implementation dependent */

        default:

            goto die;

        }

        /* Stop translation as we may have switched the execution mode */

        ctx->bstate = BS_STOP;

        break;

    case 12:

        switch (sel) {

        case 0:

            gen_op_mtc0_status();

            /* BS_STOP isn't good enough here, hflags may have changed. */

            gen_save_pc(ctx->pc + 4);

            ctx->bstate = BS_EXCP;

            rn = "Status";

            break;

        case 1:

            gen_op_mtc0_intctl();

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            rn = "IntCtl";

            break;

        case 2:

            gen_op_mtc0_srsctl();

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            rn = "SRSCtl";

            break;

        case 3:

            gen_op_mtc0_srsmap();

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            rn = "SRSMap";

            break;

        default:

            goto die;

        }

        break;

    case 13:

        switch (sel) {

        case 0:

            gen_op_mtc0_cause();

            rn = "Cause";

            break;

        default:

            goto die;

        }

        /* Stop translation as we may have switched the execution mode */

        ctx->bstate = BS_STOP;

        break;

    case 14:

        switch (sel) {

        case 0:

            gen_op_mtc0_epc();

            rn = "EPC";

            break;

        default:

            goto die;

        }

        break;

    case 15:

        switch (sel) {

        case 0:

            /* ignored */

            rn = "PRid";

            break;

        case 1:

            gen_op_mtc0_ebase();

            rn = "EBase";

            break;

        default:

            goto die;

        }

        break;

    case 16:

        switch (sel) {

        case 0:

            gen_op_mtc0_config0();

            rn = "Config";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            break;

        case 1:

            /* ignored, read only */

            rn = "Config1";

            break;

        case 2:

            gen_op_mtc0_config2();

            rn = "Config2";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            break;

        case 3:

            /* ignored, read only */

            rn = "Config3";

            break;

        /* 4,5 are reserved */

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

            goto die;

        }

        break;

    case 17:

        switch (sel) {

        case 0:

            /* ignored */

            rn = "LLAddr";

            break;

        default:

            goto die;

        }

        break;

    case 18:

        switch (sel) {

        case 0 ... 7:

            gen_op_mtc0_watchlo(sel);

            rn = "WatchLo";

            break;

        default:

            goto die;

        }

        break;

    case 19:

        switch (sel) {

        case 0 ... 7:

            gen_op_mtc0_watchhi(sel);

            rn = "WatchHi";

            break;

        default:

            goto die;

        }

        break;

    case 20:

        switch (sel) {

        case 0:

#ifdef TARGET_MIPS64

            gen_op_mtc0_xcontext();

            rn = "XContext";

            break;

#endif

        default:

            goto die;

        }

        break;

    case 21:

       /* Officially reserved, but sel 0 is used for R1x000 framemask */

        switch (sel) {

        case 0:

            gen_op_mtc0_framemask();

            rn = "Framemask";

            break;

        default:

            goto die;

        }

        break;

    case 22:

        /* ignored */

        rn = "Diagnostic"; /* implementation dependent */

        break;

    case 23:

        switch (sel) {

        case 0:

            gen_op_mtc0_debug(); /* EJTAG support */

            /* BS_STOP isn't good enough here, hflags may have changed. */

            gen_save_pc(ctx->pc + 4);

            ctx->bstate = BS_EXCP;

            rn = "Debug";

            break;

        case 1:

//            gen_op_mtc0_tracecontrol(); /* PDtrace support */

            rn = "TraceControl";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            break;

        case 2:

//            gen_op_mtc0_tracecontrol2(); /* PDtrace support */

            rn = "TraceControl2";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            break;

        case 3:

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            gen_op_mtc0_usertracedata(); /* PDtrace support */

            rn = "UserTraceData";

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

//            break;

        case 4:

//            gen_op_mtc0_debug(); /* PDtrace support */

            /* Stop translation as we may have switched the execution mode */

            ctx->bstate = BS_STOP;

            rn = "TraceBPC";

//            break;

        default:

            goto die;

        }

        break;

    case 24:

        switch (sel) {

        case 0:

            gen_op_mtc0_depc(); /* EJTAG support */

            rn = "DEPC";

            break;

        default:

            goto die;

        }

        break;

    case 25:

        switch (sel) {

        case 0:

            gen_op_mtc0_performance0();

            rn = "Performance0";

            break;

        case 1:

//            gen_op_mtc0_performance1();

            rn = "Performance1";

//            break;

        case 2:

//            gen_op_mtc0_performance2();

            rn = "Performance2";

//            break;

        case 3:

//            gen_op_mtc0_performance3();

            rn = "Performance3";

//            break;

        case 4:

//            gen_op_mtc0_performance4();

            rn = "Performance4";

//            break;

        case 5:

//            gen_op_mtc0_performance5();

            rn = "Performance5";

//            break;

        case 6:

//            gen_op_mtc0_performance6();

            rn = "Performance6";

//            break;

        case 7:

//            gen_op_mtc0_performance7();

            rn = "Performance7";

//            break;

        default:

            goto die;

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

            goto die;

        }

       break;

    case 28:

        switch (sel) {

        case 0:

        case 2:

        case 4:

        case 6:

            gen_op_mtc0_taglo();

            rn = "TagLo";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_op_mtc0_datalo();

            rn = "DataLo";

            break;

        default:

            goto die;

        }

        break;

    case 29:

        switch (sel) {

        case 0:

        case 2:

        case 4:

        case 6:

            gen_op_mtc0_taghi();

            rn = "TagHi";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_op_mtc0_datahi();

            rn = "DataHi";

            break;

        default:

            rn = "invalid sel";

            goto die;

        }

       break;

    case 30:

        switch (sel) {

        case 0:

            gen_op_mtc0_errorepc();

            rn = "ErrorEPC";

            break;

        default:

            goto die;

        }

        break;

    case 31:

        switch (sel) {

        case 0:

            gen_op_mtc0_desave(); /* EJTAG support */

            rn = "DESAVE";

            break;

        default:

            goto die;

        }

        /* Stop translation as we may have switched the execution mode */

        ctx->bstate = BS_STOP;

        break;

    default:

       goto die;

    }

#if defined MIPS_DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_IN_ASM) {

        fprintf(logfile, "mtc0 %s (reg %d sel %d)\n",

                rn, reg, sel);

    }

#endif

    return;



die:

#if defined MIPS_DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_IN_ASM) {

        fprintf(logfile, "mtc0 %s (reg %d sel %d)\n",

                rn, reg, sel);

    }

#endif

    generate_exception(ctx, EXCP_RI);

}
