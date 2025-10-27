static int mm_start_timer(struct qemu_alarm_timer *t)

{

    TIMECAPS tc;



    memset(&tc, 0, sizeof(tc));

    timeGetDevCaps(&tc, sizeof(tc));



    mm_period = tc.wPeriodMin;

    timeBeginPeriod(mm_period);



    mm_timer = timeSetEvent(1,                  /* interval (ms) */

                            mm_period,          /* resolution */

                            mm_alarm_handler,   /* function */

                            (DWORD_PTR)t,       /* parameter */

                            TIME_ONESHOT | TIME_CALLBACK_FUNCTION);



    if (!mm_timer) {

        fprintf(stderr, "Failed to initialize win32 alarm timer: %ld\n",

                GetLastError());

        timeEndPeriod(mm_period);

        return -1;

    }



    return 0;

}
