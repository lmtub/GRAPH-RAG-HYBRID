static int vty_getchars(VIOsPAPRDevice *sdev, uint8_t *buf, int max)
{
    VIOsPAPRVTYDevice *dev = VIO_SPAPR_VTY_DEVICE(sdev);
    int n = 0;
    while ((n < max) && (dev->out != dev->in)) {
        buf[n++] = dev->buf[dev->out++ % VTERM_BUFSIZE];
    qemu_chr_fe_accept_input(&dev->chardev);
    return n;