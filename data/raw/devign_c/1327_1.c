static int parse_times(void *log_ctx, int64_t **times, int *nb_times,

                       const char *times_str)

{

    char *p;

    int i, ret = 0;

    char *times_str1 = av_strdup(times_str);

    char *saveptr = NULL;



    if (!times_str1)

        return AVERROR(ENOMEM);



#define FAIL(err) ret = err; goto end



    *nb_times = 1;

    for (p = times_str1; *p; p++)

        if (*p == ',')

            (*nb_times)++;



    *times = av_malloc(sizeof(**times) * *nb_times);

    if (!*times) {

        av_log(log_ctx, AV_LOG_ERROR, "Could not allocate forced times array\n");

        FAIL(AVERROR(ENOMEM));

    }



    p = times_str1;

    for (i = 0; i < *nb_times; i++) {

        int64_t t;

        char *tstr = av_strtok(p, ",", &saveptr);

        av_assert0(tstr);

        p = NULL;



        ret = av_parse_time(&t, tstr, 1);

        if (ret < 0) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Invalid time duration specification in %s\n", p);

            FAIL(AVERROR(EINVAL));

        }

        (*times)[i] = t;



        /* check on monotonicity */

        if (i && (*times)[i-1] > (*times)[i]) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Specified time %f is greater than the following time %f\n",

                   (float)((*times)[i])/1000000, (float)((*times)[i-1])/1000000);

            FAIL(AVERROR(EINVAL));

        }

    }



end:

    av_free(times_str1);

    return ret;

}
