void vty_putchars(VIOsPAPRDevice *sdev, uint8_t *buf, int len)

{

    VIOsPAPRVTYDevice *dev = VIO_SPAPR_VTY_DEVICE(sdev);



    /* FIXME: should check the qemu_chr_fe_write() return value */

    qemu_chr_fe_write(dev->chardev, buf, len);

}
