static int ftp_passive_mode(FTPContext *s)

{

    char *res = NULL, *start, *end;

    int i;

    const char *command = "PASV\r\n";

    const int pasv_codes[] = {227, 501, 0}; /* 501 is incorrect code */



    if (ftp_send_command(s, command, pasv_codes, &res) != 227 || !res)

        goto fail;



    start = NULL;

    for (i = 0; i < strlen(res); ++i) {

        if (res[i] == '(') {

            start = res + i + 1;

        } else if (res[i] == ')') {

            end = res + i;

            break;

        }

    }

    if (!start || !end)

        goto fail;



    *end  = '\0';

    /* skip ip */

    if (!av_strtok(start, ",", &end)) goto fail;

    if (!av_strtok(end, ",", &end)) goto fail;

    if (!av_strtok(end, ",", &end)) goto fail;

    if (!av_strtok(end, ",", &end)) goto fail;



    /* parse port number */

    start = av_strtok(end, ",", &end);

    if (!start) goto fail;

    s->server_data_port = atoi(start) * 256;

    start = av_strtok(end, ",", &end);

    if (!start) goto fail;

    s->server_data_port += atoi(start);

    av_dlog(s, "Server data port: %d\n", s->server_data_port);



    av_free(res);

    return 0;



  fail:

    av_free(res);

    s->server_data_port = -1;

    return AVERROR(EIO);

}
