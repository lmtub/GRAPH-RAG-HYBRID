static int get_blocksize(BlockDriverState *bdrv)

{

    uint8_t cmd[10];

    uint8_t buf[8];

    uint8_t sensebuf[8];

    sg_io_hdr_t io_header;

    int ret;



    memset(cmd, 0, sizeof(cmd));

    memset(buf, 0, sizeof(buf));

    cmd[0] = READ_CAPACITY;



    memset(&io_header, 0, sizeof(io_header));

    io_header.interface_id = 'S';

    io_header.dxfer_direction = SG_DXFER_FROM_DEV;

    io_header.dxfer_len = sizeof(buf);

    io_header.dxferp = buf;

    io_header.cmdp = cmd;

    io_header.cmd_len = sizeof(cmd);

    io_header.mx_sb_len = sizeof(sensebuf);

    io_header.sbp = sensebuf;

    io_header.timeout = 6000; /* XXX */



    ret = bdrv_ioctl(bdrv, SG_IO, &io_header);

    if (ret < 0)

        return -1;



    return (buf[4] << 24) | (buf[5] << 16) | (buf[6] << 8) | buf[7];

}
