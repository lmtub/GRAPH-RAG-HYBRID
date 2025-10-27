static int gd_vc_chr_write(CharDriverState *chr, const uint8_t *buf, int len)

{

    VirtualConsole *vc = chr->opaque;



    return vc ? write(vc->fd, buf, len) : len;

}
