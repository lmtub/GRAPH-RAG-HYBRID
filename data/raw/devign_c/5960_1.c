static void cmd_inquiry(IDEState *s, uint8_t *buf)

{

    int max_len = buf[4];



    buf[0] = 0x05; /* CD-ROM */

    buf[1] = 0x80; /* removable */

    buf[2] = 0x00; /* ISO */

    buf[3] = 0x21; /* ATAPI-2 (XXX: put ATAPI-4 ?) */

    buf[4] = 31; /* additional length */

    buf[5] = 0; /* reserved */

    buf[6] = 0; /* reserved */

    buf[7] = 0; /* reserved */

    padstr8(buf + 8, 8, "QEMU");

    padstr8(buf + 16, 16, "QEMU DVD-ROM");

    padstr8(buf + 32, 4, s->version);

    ide_atapi_cmd_reply(s, 36, max_len);

}
