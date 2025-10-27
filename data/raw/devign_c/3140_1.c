static int init_report(const char *env)

{

    const char *filename_template = "%p-%t.log";

    char *key, *val;

    int ret, count = 0;

    time_t now;

    struct tm *tm;

    AVBPrint filename;



    if (report_file) /* already opened */

        return 0;

    time(&now);

    tm = localtime(&now);



    while (env && *env) {

        if ((ret = av_opt_get_key_value(&env, "=", ":", 0, &key, &val)) < 0) {

            if (count)

                av_log(NULL, AV_LOG_ERROR,

                       "Failed to parse FFREPORT environment variable: %s\n",

                       av_err2str(ret));

            break;

        }

        if (*env)

            env++;

        count++;

        if (!strcmp(key, "file")) {

            filename_template = val;

            val = NULL;

        } else {

            av_log(NULL, AV_LOG_ERROR, "Unknown key '%s' in FFREPORT\n", key);

        }

        av_free(val);

        av_free(key);

    }



    av_bprint_init(&filename, 0, 1);

    expand_filename_template(&filename, filename_template, tm);

    if (!av_bprint_is_complete(&filename)) {

        av_log(NULL, AV_LOG_ERROR, "Out of memory building report file name\n");

        return AVERROR(ENOMEM);

    }



    report_file = fopen(filename.str, "w");

    if (!report_file) {

        av_log(NULL, AV_LOG_ERROR, "Failed to open report \"%s\": %s\n",

               filename.str, strerror(errno));

        return AVERROR(errno);

    }

    av_log_set_callback(log_callback_report);

    av_log(NULL, AV_LOG_INFO,

           "%s started on %04d-%02d-%02d at %02d:%02d:%02d\n"

           "Report written to \"%s\"\n",

           program_name,

           tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,

           tm->tm_hour, tm->tm_min, tm->tm_sec,

           filename.str);

    av_log_set_level(FFMAX(av_log_get_level(), AV_LOG_VERBOSE));

    av_bprint_finalize(&filename, NULL);

    return 0;

}
