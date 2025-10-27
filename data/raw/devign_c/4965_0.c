static int ftp_status(FTPContext *s, char **line, const int response_codes[])

{

    int err, i, result = 0, pref_code_found = 0, wait_count = 100;

    char buf[CONTROL_BUFFER_SIZE];



    /* Set blocking mode */

    s->conn_control_block_flag = 0;

    for (;;) {

        if ((err = ftp_get_line(s, buf, sizeof(buf))) < 0) {

            if (err == AVERROR_EXIT) {

                if (!pref_code_found && wait_count--) {

                    av_usleep(10000);

                    continue;

                }

            }

            return result;

        }



        av_log(s, AV_LOG_DEBUG, "%s\n", buf);



        if (!pref_code_found) {

            if (strlen(buf) < 3)

                continue;



            err = 0;

            for (i = 0; i < 3; ++i) {

                if (buf[i] < '0' || buf[i] > '9')

                    continue;

                err *= 10;

                err += buf[i] - '0';

            }



            for (i = 0; response_codes[i]; ++i) {

                if (err == response_codes[i]) {

                    /* first code received. Now get all lines in non blocking mode */

                    s->conn_control_block_flag = 1;

                    pref_code_found = 1;

                    result = err;

                    if (line)

                        *line = av_strdup(buf);

                    break;

                }

            }

        }

    }

    return result;

}
