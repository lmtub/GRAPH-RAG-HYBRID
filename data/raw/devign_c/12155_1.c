int qemu_chr_fe_write_all(CharDriverState *s, const uint8_t *buf, int len)

{

    int offset = 0;

    int res;



    qemu_mutex_lock(&s->chr_write_lock);

    while (offset < len) {

        do {

            res = s->chr_write(s, buf + offset, len - offset);

            if (res == -1 && errno == EAGAIN) {

                g_usleep(100);

            }

        } while (res == -1 && errno == EAGAIN);



        if (res <= 0) {

            break;

        }



        offset += res;

    }

    qemu_mutex_unlock(&s->chr_write_lock);



    if (res < 0) {

        return res;

    }

    return offset;

}
