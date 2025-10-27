static ExitStatus gen_mtpr(DisasContext *ctx, TCGv vb, int regno)

{

    TCGv tmp;

    int data;



    switch (regno) {

    case 255:

        /* TBIA */

        gen_helper_tbia(cpu_env);

        break;



    case 254:

        /* TBIS */

        gen_helper_tbis(cpu_env, vb);

        break;



    case 253:

        /* WAIT */

        tmp = tcg_const_i64(1);

        tcg_gen_st32_i64(tmp, cpu_env, -offsetof(AlphaCPU, env) +

                                       offsetof(CPUState, halted));

        return gen_excp(ctx, EXCP_HALTED, 0);



    case 252:

        /* HALT */

        gen_helper_halt(vb);

        return EXIT_PC_STALE;



    case 251:

        /* ALARM */

        gen_helper_set_alarm(cpu_env, vb);

        break;



    case 7:

        /* PALBR */

        tcg_gen_st_i64(vb, cpu_env, offsetof(CPUAlphaState, palbr));

        /* Changing the PAL base register implies un-chaining all of the TBs

           that ended with a CALL_PAL.  Since the base register usually only

           changes during boot, flushing everything works well.  */

        gen_helper_tb_flush(cpu_env);

        return EXIT_PC_STALE;



    case 32 ... 39:

        /* Accessing the "non-shadow" general registers.  */

        regno = regno == 39 ? 25 : regno - 32 + 8;

        tcg_gen_mov_i64(cpu_std_ir[regno], vb);

        break;



    case 0: /* PS */

        st_flag_byte(vb, ENV_FLAG_PS_SHIFT);

        break;

    case 1: /* FEN */

        st_flag_byte(vb, ENV_FLAG_FEN_SHIFT);

        break;



    default:

        /* The basic registers are data only, and unknown registers

           are read-zero, write-ignore.  */

        data = cpu_pr_data(regno);

        if (data != 0) {

            if (data & PR_LONG) {

                tcg_gen_st32_i64(vb, cpu_env, data & ~PR_LONG);

            } else {

                tcg_gen_st_i64(vb, cpu_env, data);

            }

        }

        break;

    }



    return NO_EXIT;

}
