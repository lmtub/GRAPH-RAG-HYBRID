int av_parse_time(int64_t *timeval, const char *timestr, int duration)

{

    const char *p;

    int64_t t;

    struct tm dt = { 0 };

    int i;

    static const char * const date_fmt[] = {

        "%Y-%m-%d",

        "%Y%m%d",

    };

    static const char * const time_fmt[] = {

        "%H:%M:%S",

        "%H%M%S",

    };

    const char *q;

    int is_utc, len;

    char lastch;

    int negative = 0;



#undef time

    time_t now = time(0);



    len = strlen(timestr);

    if (len > 0)

        lastch = timestr[len - 1];

    else

        lastch = '\0';

    is_utc = (lastch == 'z' || lastch == 'Z');



    p = timestr;

    q = NULL;

    if (!duration) {

        if (!av_strncasecmp(timestr, "now", len)) {

            *timeval = (int64_t) now * 1000000;

            return 0;

        }



        /* parse the year-month-day part */

        for (i = 0; i < FF_ARRAY_ELEMS(date_fmt); i++) {

            q = small_strptime(p, date_fmt[i], &dt);

            if (q) {

                break;

            }

        }



        /* if the year-month-day part is missing, then take the

         * current year-month-day time */

        if (!q) {

            if (is_utc) {

                dt = *gmtime(&now);

            } else {

                dt = *localtime(&now);

            }

            dt.tm_hour = dt.tm_min = dt.tm_sec = 0;

        } else {

            p = q;

        }



        if (*p == 'T' || *p == 't' || *p == ' ')

            p++;



        /* parse the hour-minute-second part */

        for (i = 0; i < FF_ARRAY_ELEMS(time_fmt); i++) {

            q = small_strptime(p, time_fmt[i], &dt);

            if (q) {

                break;

            }

        }

    } else {

        /* parse timestr as a duration */

        if (p[0] == '-') {

            negative = 1;

            ++p;

        }

        /* parse timestr as HH:MM:SS */

        q = small_strptime(p, time_fmt[0], &dt);

        if (!q) {

            /* parse timestr as S+ */

            dt.tm_sec = strtol(p, (void *)&q, 10);

            if (q == p) {

                /* the parsing didn't succeed */

                *timeval = INT64_MIN;

                return AVERROR(EINVAL);

            }

            dt.tm_min = 0;

            dt.tm_hour = 0;

        }

    }



    /* Now we have all the fields that we can get */

    if (!q) {

        *timeval = INT64_MIN;

        return AVERROR(EINVAL);

    }



    if (duration) {

        t = dt.tm_hour * 3600 + dt.tm_min * 60 + dt.tm_sec;

    } else {

        dt.tm_isdst = -1;       /* unknown */

        if (is_utc) {

            t = av_timegm(&dt);

        } else {

            t = mktime(&dt);

        }

    }



    t *= 1000000;



    /* parse the .m... part */

    if (*q == '.') {

        int val, n;

        q++;

        for (val = 0, n = 100000; n >= 1; n /= 10, q++) {

            if (!isdigit(*q))

                break;

            val += n * (*q - '0');

        }

        t += val;

    }

    *timeval = negative ? -t : t;

    return 0;

}
