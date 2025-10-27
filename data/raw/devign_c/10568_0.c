static void gen_dmfc0 (DisasContext *ctx, int reg, int sel)

{

    const char *rn = "invalid";



    switch (reg) {

    case 0:

        switch (sel) {

        case 0:

           gen_op_mfc0_index();

            rn = "Index";

            break;

        case 1:

//         gen_op_dmfc0_mvpcontrol(); /* MT ASE */

            rn = "MVPControl";

//         break;

        case 2:

//         gen_op_dmfc0_mvpconf0(); /* MT ASE */

            rn = "MVPConf0";

//         break;

        case 3:

//         gen_op_dmfc0_mvpconf1(); /* MT ASE */

            rn = "MVPConf1";

//         break;

        default:

            goto die;

        }

        break;

    case 1:

        switch (sel) {

        case 0:

            gen_op_mfc0_random();

            rn = "Random";

           break;

        case 1:

//         gen_op_dmfc0_vpecontrol(); /* MT ASE */

            rn = "VPEControl";

//         break;

        case 2:

//         gen_op_dmfc0_vpeconf0(); /* MT ASE */

            rn = "VPEConf0";

//         break;

        case 3:

//         gen_op_dmfc0_vpeconf1(); /* MT ASE */

            rn = "VPEConf1";

//         break;

        case 4:

//         gen_op_dmfc0_YQMask(); /* MT ASE */

            rn = "YQMask";

//         break;

        case 5:

//         gen_op_dmfc0_vpeschedule(); /* MT ASE */

            rn = "VPESchedule";

//         break;

        case 6:

//         gen_op_dmfc0_vpeschefback(); /* MT ASE */

            rn = "VPEScheFBack";

//         break;

        case 7:

//         gen_op_dmfc0_vpeopt(); /* MT ASE */

            rn = "VPEOpt";

//         break;

        default:

            goto die;

        }

        break;

    case 2:

        switch (sel) {

        case 0:

           gen_op_dmfc0_entrylo0();

           rn = "EntryLo0";

           break;

        case 1:

//         gen_op_dmfc0_tcstatus(); /* MT ASE */

           rn = "TCStatus";

//         break;

        case 2:

//         gen_op_dmfc0_tcbind(); /* MT ASE */

           rn = "TCBind";

//         break;

        case 3:

//         gen_op_dmfc0_tcrestart(); /* MT ASE */

           rn = "TCRestart";

//         break;

        case 4:

//         gen_op_dmfc0_tchalt(); /* MT ASE */

           rn = "TCHalt";

//         break;

        case 5:

//         gen_op_dmfc0_tccontext(); /* MT ASE */

           rn = "TCContext";

//         break;

        case 6:

//         gen_op_dmfc0_tcschedule(); /* MT ASE */

           rn = "TCSchedule";

//         break;

        case 7:

//         gen_op_dmfc0_tcschefback(); /* MT ASE */

           rn = "TCScheFBack";

//         break;

        default:

            goto die;

        }

        break;

    case 3:

        switch (sel) {

        case 0:

           gen_op_dmfc0_entrylo1();

           rn = "EntryLo1";

           break;

        default:

            goto die;

       }

        break;

    case 4:

        switch (sel) {

        case 0:

           gen_op_dmfc0_context();

           rn = "Context";

           break;

        case 1:

//         gen_op_dmfc0_contextconfig(); /* SmartMIPS ASE */

           rn = "ContextConfig";

//         break;

        default:

            goto die;

       }

        break;

    case 5:

        switch (sel) {

        case 0:

           gen_op_mfc0_pagemask();

           rn = "PageMask";

           break;

        case 1:

           gen_op_mfc0_pagegrain();

           rn = "PageGrain";

           break;

        default:

            goto die;

       }

        break;

    case 6:

        switch (sel) {

        case 0:

           gen_op_mfc0_wired();

           rn = "Wired";

           break;

        case 1:

//         gen_op_dmfc0_srsconf0(); /* shadow registers */

           rn = "SRSConf0";

//         break;

        case 2:

//         gen_op_dmfc0_srsconf1(); /* shadow registers */

           rn = "SRSConf1";

//         break;

        case 3:

//         gen_op_dmfc0_srsconf2(); /* shadow registers */

           rn = "SRSConf2";

//         break;

        case 4:

//         gen_op_dmfc0_srsconf3(); /* shadow registers */

           rn = "SRSConf3";

//         break;

        case 5:

//         gen_op_dmfc0_srsconf4(); /* shadow registers */

           rn = "SRSConf4";

//         break;

        default:

            goto die;

       }

        break;

    case 7:

        switch (sel) {

        case 0:

           gen_op_mfc0_hwrena();

           rn = "HWREna";

           break;

        default:

            goto die;

       }

        break;

    case 8:

        switch (sel) {

        case 0:

           gen_op_dmfc0_badvaddr();

           rn = "BadVaddr";

           break;

        default:

            goto die;

       }

        break;

    case 9:

        switch (sel) {

        case 0:

           gen_op_mfc0_count();

           rn = "Count";

           break;

       /* 6,7 are implementation dependent */

        default:

            goto die;

       }

        break;

    case 10:

        switch (sel) {

        case 0:

           gen_op_dmfc0_entryhi();

           rn = "EntryHi";

           break;

        default:

            goto die;

       }

        break;

    case 11:

        switch (sel) {

        case 0:

           gen_op_mfc0_compare();

           rn = "Compare";

           break;

       /* 6,7 are implementation dependent */

        default:

            goto die;

       }

        break;

    case 12:

        switch (sel) {

        case 0:

           gen_op_mfc0_status();

           rn = "Status";

           break;

        case 1:

           gen_op_mfc0_intctl();

           rn = "IntCtl";

           break;

        case 2:

           gen_op_mfc0_srsctl();

           rn = "SRSCtl";

           break;

        case 3:

           gen_op_mfc0_srsmap(); /* shadow registers */

           rn = "SRSMap";

           break;

        default:

            goto die;

       }

        break;

    case 13:

        switch (sel) {

        case 0:

           gen_op_mfc0_cause();

           rn = "Cause";

           break;

        default:

            goto die;

       }

        break;

    case 14:

        switch (sel) {

        case 0:

           gen_op_dmfc0_epc();

           rn = "EPC";

           break;

        default:

            goto die;

       }

        break;

    case 15:

        switch (sel) {

        case 0:

           gen_op_mfc0_prid();

           rn = "PRid";

           break;

        case 1:

           gen_op_dmfc0_ebase();

           rn = "EBase";

           break;

        default:

            goto die;

       }

        break;

    case 16:

        switch (sel) {

        case 0:

           gen_op_mfc0_config0();

            rn = "Config";

            break;

        case 1:

           gen_op_mfc0_config1();

            rn = "Config1";

            break;

        case 2:

           gen_op_mfc0_config2();

            rn = "Config2";

            break;

        case 3:

           gen_op_mfc0_config3();

            rn = "Config3";

            break;

       /* 6,7 are implementation dependent */

        default:

            goto die;

        }

        break;

    case 17:

        switch (sel) {

        case 0:

           gen_op_dmfc0_lladdr();

           rn = "LLAddr";

           break;

        default:

            goto die;

        }

        break;

    case 18:

        switch (sel) {

        case 0:

           gen_op_dmfc0_watchlo0();

           rn = "WatchLo";

           break;

        case 1:

//         gen_op_dmfc0_watchlo1();

           rn = "WatchLo1";

//         break;

        case 2:

//         gen_op_dmfc0_watchlo2();

           rn = "WatchLo2";

//         break;

        case 3:

//         gen_op_dmfc0_watchlo3();

           rn = "WatchLo3";

//         break;

        case 4:

//         gen_op_dmfc0_watchlo4();

           rn = "WatchLo4";

//         break;

        case 5:

//         gen_op_dmfc0_watchlo5();

           rn = "WatchLo5";

//         break;

        case 6:

//         gen_op_dmfc0_watchlo6();

           rn = "WatchLo6";

//         break;

        case 7:

//         gen_op_dmfc0_watchlo7();

           rn = "WatchLo7";

//         break;

        default:

            goto die;

        }

        break;

    case 19:

        switch (sel) {

        case 0:

           gen_op_mfc0_watchhi0();

           rn = "WatchHi";

           break;

        case 1:

//         gen_op_mfc0_watchhi1();

           rn = "WatchHi1";

//         break;

        case 2:

//         gen_op_mfc0_watchhi2();

           rn = "WatchHi2";

//         break;

        case 3:

//         gen_op_mfc0_watchhi3();

           rn = "WatchHi3";

//         break;

        case 4:

//         gen_op_mfc0_watchhi4();

           rn = "WatchHi4";

//         break;

        case 5:

//         gen_op_mfc0_watchhi5();

           rn = "WatchHi5";

//         break;

        case 6:

//         gen_op_mfc0_watchhi6();

           rn = "WatchHi6";

//         break;

        case 7:

//         gen_op_mfc0_watchhi7();

           rn = "WatchHi7";

//         break;

        default:

            goto die;

        }

        break;

    case 20:

        switch (sel) {

        case 0:

           /* 64 bit MMU only */

           gen_op_dmfc0_xcontext();

           rn = "XContext";

           break;

        default:

            goto die;

        }

        break;

    case 21:

       /* Officially reserved, but sel 0 is used for R1x000 framemask */

        switch (sel) {

        case 0:

           gen_op_mfc0_framemask();

           rn = "Framemask";

           break;

        default:

            goto die;

        }

        break;

    case 22:

       /* ignored */

       rn = "'Diagnostic"; /* implementation dependent */

       break;

    case 23:

        switch (sel) {

        case 0:

           gen_op_mfc0_debug(); /* EJTAG support */

           rn = "Debug";

           break;

        case 1:

//         gen_op_dmfc0_tracecontrol(); /* PDtrace support */

           rn = "TraceControl";

//         break;

        case 2:

//         gen_op_dmfc0_tracecontrol2(); /* PDtrace support */

           rn = "TraceControl2";

//         break;

        case 3:

//         gen_op_dmfc0_usertracedata(); /* PDtrace support */

           rn = "UserTraceData";

//         break;

        case 4:

//         gen_op_dmfc0_debug(); /* PDtrace support */

           rn = "TraceBPC";

//         break;

        default:

            goto die;

        }

        break;

    case 24:

        switch (sel) {

        case 0:

           gen_op_dmfc0_depc(); /* EJTAG support */

           rn = "DEPC";

           break;

        default:

            goto die;

        }

        break;

    case 25:

        switch (sel) {

        case 0:

           gen_op_mfc0_performance0();

           rn = "Performance0";

            break;

        case 1:

//         gen_op_dmfc0_performance1();

           rn = "Performance1";

//         break;

        case 2:

//         gen_op_dmfc0_performance2();

           rn = "Performance2";

//         break;

        case 3:

//         gen_op_dmfc0_performance3();

           rn = "Performance3";

//         break;

        case 4:

//         gen_op_dmfc0_performance4();

           rn = "Performance4";

//         break;

        case 5:

//         gen_op_dmfc0_performance5();

           rn = "Performance5";

//         break;

        case 6:

//         gen_op_dmfc0_performance6();

           rn = "Performance6";

//         break;

        case 7:

//         gen_op_dmfc0_performance7();

           rn = "Performance7";

//         break;

        default:

            goto die;

        }

        break;

    case 26:

       rn = "ECC";

       break;

    case 27:

        switch (sel) {

        /* ignored */

        case 0 ... 3:

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

            gen_op_mfc0_taglo();

            rn = "TagLo";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_op_mfc0_datalo();

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

            gen_op_mfc0_taghi();

            rn = "TagHi";

            break;

        case 1:

        case 3:

        case 5:

        case 7:

            gen_op_mfc0_datahi();

            rn = "DataHi";

            break;

        default:

            goto die;

        }

        break;

    case 30:

        switch (sel) {

        case 0:

           gen_op_dmfc0_errorepc();

           rn = "ErrorEPC";

           break;

        default:

            goto die;

        }

        break;

    case 31:

        switch (sel) {

        case 0:

           gen_op_mfc0_desave(); /* EJTAG support */

           rn = "DESAVE";

           break;

        default:

            goto die;

        }

        break;

    default:

       goto die;

    }

#if defined MIPS_DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_IN_ASM) {

        fprintf(logfile, "dmfc0 %s (reg %d sel %d)\n",

                rn, reg, sel);

    }

#endif

    return;



die:

#if defined MIPS_DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_IN_ASM) {

        fprintf(logfile, "dmfc0 %s (reg %d sel %d)\n",

                rn, reg, sel);

    }

#endif

    generate_exception(ctx, EXCP_RI);

}
