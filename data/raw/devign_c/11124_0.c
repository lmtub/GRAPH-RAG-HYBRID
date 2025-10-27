void qemu_get_timedate(struct tm *tm, int offset)

{

    time_t ti;



    time(&ti);

    ti += offset;

    if (rtc_date_offset == -1) {

        if (rtc_utc)

            gmtime_r(&ti, tm);

        else

            localtime_r(&ti, tm);

    } else {

        ti -= rtc_date_offset;

        gmtime_r(&ti, tm);

    }

}
