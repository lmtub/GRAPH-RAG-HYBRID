int qemu_chr_fe_ioctl(CharDriverState *s, int cmd, void *arg)

{

    if (!s->chr_ioctl)

        return -ENOTSUP;

    return s->chr_ioctl(s, cmd, arg);

}
