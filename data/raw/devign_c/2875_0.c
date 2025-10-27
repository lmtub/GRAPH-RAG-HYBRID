int qemu_timedate_diff(struct tm *tm)

{

    time_t seconds;



    if (rtc_date_offset == -1)

        if (rtc_utc)

            seconds = mktimegm(tm);

        else {

            struct tm tmp = *tm;

            tmp.tm_isdst = -1; /* use timezone to figure it out */

            seconds = mktime(&tmp);

	}

    else

        seconds = mktimegm(tm) + rtc_date_offset;



    return seconds - time(NULL);

}
