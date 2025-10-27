static void tcp_chr_telnet_init(QIOChannel *ioc)

{

    char buf[3];

    /* Send the telnet negotion to put telnet in binary, no echo, single char mode */

    IACSET(buf, 0xff, 0xfb, 0x01);  /* IAC WILL ECHO */

    qio_channel_write(ioc, buf, 3, NULL);

    IACSET(buf, 0xff, 0xfb, 0x03);  /* IAC WILL Suppress go ahead */

    qio_channel_write(ioc, buf, 3, NULL);

    IACSET(buf, 0xff, 0xfb, 0x00);  /* IAC WILL Binary */

    qio_channel_write(ioc, buf, 3, NULL);

    IACSET(buf, 0xff, 0xfd, 0x00);  /* IAC DO Binary */

    qio_channel_write(ioc, buf, 3, NULL);

}
