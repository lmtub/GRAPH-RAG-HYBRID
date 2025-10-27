ssize_t slirp_send(struct socket *so, const void *buf, size_t len, int flags)

{

    if (so->s == -1 && so->extra) {

        qemu_chr_fe_write(so->extra, buf, len);

        return len;

    }



    return send(so->s, buf, len, flags);

}
