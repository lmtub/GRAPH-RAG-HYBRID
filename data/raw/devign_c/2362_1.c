static int64_t ivshmem_recv_msg(IVShmemState *s, int *pfd, Error **errp)

{

    int64_t msg;

    int n, ret;



    n = 0;

    do {

        ret = qemu_chr_fe_read_all(&s->server_chr, (uint8_t *)&msg + n,

                                   sizeof(msg) - n);

        if (ret < 0) {

            if (ret == -EINTR) {

                continue;

            }

            error_setg_errno(errp, -ret, "read from server failed");

            return INT64_MIN;

        }

        n += ret;

    } while (n < sizeof(msg));



    *pfd = qemu_chr_fe_get_msgfd(&s->server_chr);

    return msg;

}
