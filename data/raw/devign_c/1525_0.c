static int ftp_file_size(FTPContext *s)

{

    char command[CONTROL_BUFFER_SIZE];

    char *res = NULL;

    const int size_codes[] = {213, 0};



    snprintf(command, sizeof(command), "SIZE %s\r\n", s->path);

    if (ftp_send_command(s, command, size_codes, &res)) {

        s->filesize = strtoll(&res[4], NULL, 10);

    } else {

        s->filesize = -1;

        av_free(res);

        return AVERROR(EIO);

    }



    av_free(res);

    return 0;

}
