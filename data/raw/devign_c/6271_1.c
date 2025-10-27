int qemu_get_buffer(QEMUFile *f, uint8_t *buf, int size)

{

    int pending = size;

    int done = 0;



    while (pending > 0) {

        int res;



        res = qemu_peek_buffer(f, buf, pending, 0);

        if (res == 0) {

            return done;

        }

        qemu_file_skip(f, res);

        buf += res;

        pending -= res;

        done += res;

    }

    return done;

}
