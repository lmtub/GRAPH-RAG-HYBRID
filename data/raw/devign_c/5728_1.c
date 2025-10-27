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


        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_mvpcontrol(arg, cpu_env);

            rn = "MVPControl";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_mvpconf0(arg, cpu_env);

            rn = "MVPConf0";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_mvpconf1(arg, cpu_env);

            rn = "MVPConf1";



            CP0_CHECK(ctx->vp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPControl));

            rn = "VPControl";


        default:

            goto cp0_unimplemented;

        }


    case 1:

        switch (sel) {

        case 0:

            CP0_CHECK(!(ctx->insn_flags & ISA_MIPS32R6));

            gen_helper_mfc0_random(arg, cpu_env);

            rn = "Random";


        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEControl));

            rn = "VPEControl";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEConf0));

            rn = "VPEConf0";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEConf1));

            rn = "VPEConf1";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_YQMask));

            rn = "YQMask";


        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_VPESchedule));

            rn = "VPESchedule";


        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_VPEScheFBack));

            rn = "VPEScheFBack";


        case 7:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_VPEOpt));

            rn = "VPEOpt";


        default:

            goto cp0_unimplemented;

        }



        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EntryLo0));

            rn = "EntryLo0";


        case 1:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tcstatus(arg, cpu_env);

            rn = "TCStatus";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_mfc0_tcbind(arg, cpu_env);

            rn = "TCBind";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tcrestart(arg, cpu_env);

            rn = "TCRestart";



            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tchalt(arg, cpu_env);

            rn = "TCHalt";


        case 5:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tccontext(arg, cpu_env);

            rn = "TCContext";


        case 6:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tcschedule(arg, cpu_env);

            rn = "TCSchedule";


        case 7:

            CP0_CHECK(ctx->insn_flags & ASE_MT);

            gen_helper_dmfc0_tcschefback(arg, cpu_env);

            rn = "TCScheFBack";


        default:

            goto cp0_unimplemented;

        }



        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EntryLo1));

            rn = "EntryLo1";


        case 1:

            CP0_CHECK(ctx->vp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_GlobalNumber));

            rn = "GlobalNumber";


        default:

            goto cp0_unimplemented;

        }



        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_Context));

            rn = "Context";


        case 1:

//            gen_helper_dmfc0_contextconfig(arg); /* SmartMIPS ASE */

            rn = "ContextConfig";

            goto cp0_unimplemented;


            CP0_CHECK(ctx->ulri);

            tcg_gen_ld_tl(arg, cpu_env,

                          offsetof(CPUMIPSState, active_tc.CP0_UserLocal));

            rn = "UserLocal";


        default:

            goto cp0_unimplemented;

        }


    case 5:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_PageMask));

            rn = "PageMask";


        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_PageGrain));

            rn = "PageGrain";

















        default:

            goto cp0_unimplemented;

        }


    case 6:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Wired));

            rn = "Wired";


        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf0));

            rn = "SRSConf0";



            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf1));

            rn = "SRSConf1";



            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf2));

            rn = "SRSConf2";



            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf3));

            rn = "SRSConf3";


        case 5:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSConf4));

            rn = "SRSConf4";


        default:

            goto cp0_unimplemented;

        }


    case 7:

        switch (sel) {

        case 0:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_HWREna));

            rn = "HWREna";


        default:

            goto cp0_unimplemented;

        }


    case 8:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_BadVAddr));

            rn = "BadVAddr";


        case 1:

            CP0_CHECK(ctx->bi);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_BadInstr));

            rn = "BadInstr";



            CP0_CHECK(ctx->bp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_BadInstrP));

            rn = "BadInstrP";


        default:

            goto cp0_unimplemented;

        }


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

               after reading count.  */

            ctx->bstate = BS_STOP;

            rn = "Count";


        /* 6,7 are implementation dependent */

        default:

            goto cp0_unimplemented;

        }


    case 10:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EntryHi));

            rn = "EntryHi";


        default:

            goto cp0_unimplemented;

        }


    case 11:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Compare));

            rn = "Compare";


        /* 6,7 are implementation dependent */

        default:

            goto cp0_unimplemented;

        }


    case 12:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Status));

            rn = "Status";


        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_IntCtl));

            rn = "IntCtl";



            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSCtl));

            rn = "SRSCtl";



            check_insn(ctx, ISA_MIPS32R2);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_SRSMap));

            rn = "SRSMap";


        default:

            goto cp0_unimplemented;

        }


    case 13:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Cause));

            rn = "Cause";


        default:

            goto cp0_unimplemented;

        }


    case 14:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EPC));

            rn = "EPC";


        default:

            goto cp0_unimplemented;

        }


    case 15:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_PRid));

            rn = "PRid";


        case 1:

            check_insn(ctx, ISA_MIPS32R2);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_EBase));

            rn = "EBase";



            check_insn(ctx, ISA_MIPS32R2);

            CP0_CHECK(ctx->cmgcr);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_CMGCRBase));

            rn = "CMGCRBase";


        default:

            goto cp0_unimplemented;

        }


    case 16:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config0));

            rn = "Config";


        case 1:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config1));

            rn = "Config1";



            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config2));

            rn = "Config2";



            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config3));

            rn = "Config3";



            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config4));

            rn = "Config4";


        case 5:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config5));

            rn = "Config5";


       /* 6,7 are implementation dependent */

        case 6:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config6));

            rn = "Config6";


        case 7:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Config7));

            rn = "Config7";


        default:

            goto cp0_unimplemented;

        }


    case 17:

        switch (sel) {

        case 0:

            gen_helper_dmfc0_lladdr(arg, cpu_env);

            rn = "LLAddr";


        case 1:

            CP0_CHECK(ctx->mrp);

            gen_helper_dmfc0_maar(arg, cpu_env);

            rn = "MAAR";



            CP0_CHECK(ctx->mrp);

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_MAARI));

            rn = "MAARI";


        default:

            goto cp0_unimplemented;

        }


    case 18:

        switch (sel) {

        case 0 ... 7:

            gen_helper_1e0i(dmfc0_watchlo, arg, sel);

            rn = "WatchLo";


        default:

            goto cp0_unimplemented;

        }


    case 19:

        switch (sel) {

        case 0 ... 7:

            gen_helper_1e0i(mfc0_watchhi, arg, sel);

            rn = "WatchHi";


        default:

            goto cp0_unimplemented;

        }


    case 20:

        switch (sel) {

        case 0:

            check_insn(ctx, ISA_MIPS3);

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_XContext));

            rn = "XContext";


        default:

            goto cp0_unimplemented;

        }


    case 21:

       /* Officially reserved, but sel 0 is used for R1x000 framemask */

        CP0_CHECK(!(ctx->insn_flags & ISA_MIPS32R6));

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Framemask));

            rn = "Framemask";


        default:

            goto cp0_unimplemented;

        }


    case 22:

        tcg_gen_movi_tl(arg, 0); /* unimplemented */

        rn = "'Diagnostic"; /* implementation dependent */


    case 23:

        switch (sel) {

        case 0:

            gen_helper_mfc0_debug(arg, cpu_env); /* EJTAG support */

            rn = "Debug";


        case 1:

//            gen_helper_dmfc0_tracecontrol(arg, cpu_env); /* PDtrace support */

            rn = "TraceControl";

            goto cp0_unimplemented;


//            gen_helper_dmfc0_tracecontrol2(arg, cpu_env); /* PDtrace support */

            rn = "TraceControl2";

            goto cp0_unimplemented;


//            gen_helper_dmfc0_usertracedata(arg, cpu_env); /* PDtrace support */

            rn = "UserTraceData";

            goto cp0_unimplemented;


//            gen_helper_dmfc0_tracebpc(arg, cpu_env); /* PDtrace support */

            rn = "TraceBPC";

            goto cp0_unimplemented;

        default:

            goto cp0_unimplemented;

        }


    case 24:

        switch (sel) {

        case 0:

            /* EJTAG support */

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_DEPC));

            rn = "DEPC";


        default:

            goto cp0_unimplemented;

        }


    case 25:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_Performance0));

            rn = "Performance0";


        case 1:

//            gen_helper_dmfc0_performance1(arg);

            rn = "Performance1";

            goto cp0_unimplemented;


//            gen_helper_dmfc0_performance2(arg);

            rn = "Performance2";

            goto cp0_unimplemented;


//            gen_helper_dmfc0_performance3(arg);

            rn = "Performance3";

            goto cp0_unimplemented;


//            gen_helper_dmfc0_performance4(arg);

            rn = "Performance4";

            goto cp0_unimplemented;

        case 5:

//            gen_helper_dmfc0_performance5(arg);

            rn = "Performance5";

            goto cp0_unimplemented;

        case 6:

//            gen_helper_dmfc0_performance6(arg);

            rn = "Performance6";

            goto cp0_unimplemented;

        case 7:

//            gen_helper_dmfc0_performance7(arg);

            rn = "Performance7";

            goto cp0_unimplemented;

        default:

            goto cp0_unimplemented;

        }


    case 26:

        switch (sel) {

        case 0:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_ErrCtl));

            rn = "ErrCtl";


        default:

            goto cp0_unimplemented;

        }


    case 27:

        switch (sel) {

        /* ignored */

        case 0 ... 3:

            tcg_gen_movi_tl(arg, 0); /* unimplemented */

            rn = "CacheErr";


        default:

            goto cp0_unimplemented;

        }


    case 28:

        switch (sel) {

        case 0:



        case 6:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_TagLo));

            rn = "TagLo";


        case 1:


        case 5:

        case 7:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_DataLo));

            rn = "DataLo";


        default:

            goto cp0_unimplemented;

        }


    case 29:

        switch (sel) {

        case 0:



        case 6:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_TagHi));

            rn = "TagHi";


        case 1:


        case 5:

        case 7:

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_DataHi));

            rn = "DataHi";


        default:

            goto cp0_unimplemented;

        }


    case 30:

        switch (sel) {

        case 0:

            tcg_gen_ld_tl(arg, cpu_env, offsetof(CPUMIPSState, CP0_ErrorEPC));

            rn = "ErrorEPC";


        default:

            goto cp0_unimplemented;

        }


    case 31:

        switch (sel) {

        case 0:

            /* EJTAG support */

            gen_mfc0_load32(arg, offsetof(CPUMIPSState, CP0_DESAVE));

            rn = "DESAVE";


        case 2 ... 7:

            CP0_CHECK(ctx->kscrexist & (1 << sel));

            tcg_gen_ld_tl(arg, cpu_env,

                          offsetof(CPUMIPSState, CP0_KScratch[sel-2]));

            rn = "KScratch";


        default:

            goto cp0_unimplemented;

        }


    default:

        goto cp0_unimplemented;

    }

    trace_mips_translate_c0("dmfc0", rn, reg, sel);

    return;



cp0_unimplemented:

    qemu_log_mask(LOG_UNIMP, "dmfc0 %s (reg %d sel %d)\n", rn, reg, sel);

    gen_mfc0_unimplemented(ctx, arg);

}