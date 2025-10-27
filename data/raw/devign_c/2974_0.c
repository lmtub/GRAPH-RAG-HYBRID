static int ftp_type(FTPContext *s)

{

    const char *command = "TYPE I\r\n";

    const int type_codes[] = {200, 0};



    if (!ftp_send_command(s, command, type_codes, NULL))

        return AVERROR(EIO);



    return 0;

}
