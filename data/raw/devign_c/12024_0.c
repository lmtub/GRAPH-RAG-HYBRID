static int tap_set_sndbuf(TAPState *s, const char *sndbuf_str)

{

    int sndbuf = TAP_DEFAULT_SNDBUF;



    if (sndbuf_str) {

        sndbuf = atoi(sndbuf_str);

    }



    if (!sndbuf) {

        sndbuf = INT_MAX;

    }



    if (ioctl(s->fd, TUNSETSNDBUF, &sndbuf) == -1 && sndbuf_str) {

        qemu_error("TUNSETSNDBUF ioctl failed: %s\n", strerror(errno));

        return -1;

    }

    return 0;

}
