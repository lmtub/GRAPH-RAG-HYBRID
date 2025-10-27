static void rtas_get_time_of_day(PowerPCCPU *cpu, sPAPREnvironment *spapr,

                                 uint32_t token, uint32_t nargs,

                                 target_ulong args,

                                 uint32_t nret, target_ulong rets)

{

    struct tm tm;



    if (nret != 8) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    qemu_get_timedate(&tm, spapr->rtc_offset);



    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

    rtas_st(rets, 1, tm.tm_year + 1900);

    rtas_st(rets, 2, tm.tm_mon + 1);

    rtas_st(rets, 3, tm.tm_mday);

    rtas_st(rets, 4, tm.tm_hour);

    rtas_st(rets, 5, tm.tm_min);

    rtas_st(rets, 6, tm.tm_sec);

    rtas_st(rets, 7, 0); /* we don't do nanoseconds */

}
