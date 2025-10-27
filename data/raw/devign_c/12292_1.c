static int srt_get_duration(uint8_t **buf)

{

    int i, duration = 0;



    for (i=0; i<2 && !duration; i++) {

        int s_hour, s_min, s_sec, s_hsec, e_hour, e_min, e_sec, e_hsec;

        if (sscanf(*buf, "%d:%2d:%2d%*1[,.]%3d --> %d:%2d:%2d%*1[,.]%3d",

                   &s_hour, &s_min, &s_sec, &s_hsec,

                   &e_hour, &e_min, &e_sec, &e_hsec) == 8) {

            s_min  +=   60*s_hour;      e_min  +=   60*e_hour;

            s_sec  +=   60*s_min;       e_sec  +=   60*e_min;

            s_hsec += 1000*s_sec;       e_hsec += 1000*e_sec;

            duration = e_hsec - s_hsec;

        }

        *buf += strcspn(*buf, "\n") + 1;

    }

    return duration;

}
