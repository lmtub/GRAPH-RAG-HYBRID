static int ftp_flush_control_input(FTPContext *s)

{

    char buf[CONTROL_BUFFER_SIZE];

    int err, ori_block_flag = s->conn_control_block_flag;



    s->conn_control_block_flag = 1;

    do {

        err = ftp_get_line(s, buf, sizeof(buf));

    } while (!err);



    s->conn_control_block_flag = ori_block_flag;



    if (err < 0 && err != AVERROR_EXIT)

        return err;



    return 0;

}
