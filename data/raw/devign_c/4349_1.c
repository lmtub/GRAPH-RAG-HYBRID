static int parse_interval(Interval *interval, int interval_count,

                          const char **buf, void *log_ctx)

{

    char *intervalstr;

    int ret;



    *buf += strspn(*buf, SPACES);

    if (!**buf)

        return 0;



    /* reset data */

    memset(interval, 0, sizeof(Interval));

    interval->index = interval_count;



    /* format: INTERVAL COMMANDS */



    /* parse interval */

    intervalstr = av_get_token(buf, DELIMS);

    if (intervalstr && intervalstr[0]) {

        char *start, *end;



        start = av_strtok(intervalstr, "-", &end);








        if ((ret = av_parse_time(&interval->start_ts, start, 1)) < 0) {


                   "Invalid start time specification '%s' in interval #%d\n",

                   start, interval_count);





        if (end) {

            if ((ret = av_parse_time(&interval->end_ts, end, 1)) < 0) {


                       "Invalid end time specification '%s' in interval #%d\n",

                       end, interval_count);



        } else {

            interval->end_ts = INT64_MAX;


        if (interval->end_ts < interval->start_ts) {


                   "Invalid end time '%s' in interval #%d: "

                   "cannot be lesser than start time '%s'\n",

                   end, interval_count, start);




    } else {


               "No interval specified for interval #%d\n", interval_count);






    /* parse commands */

    ret = parse_commands(&interval->commands, &interval->nb_commands,

                         interval_count, buf, log_ctx);



end:

    av_free(intervalstr);

    return ret;
