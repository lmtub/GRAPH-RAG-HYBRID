int qemu_chr_fe_read_all(CharDriverState *s, uint8_t *buf, int len)

{

    int offset = 0, counter = 10;

    int res;



    if (!s->chr_sync_read) {

        return 0;

    }

    

    if (s->replay && replay_mode == REPLAY_MODE_PLAY) {

        return replay_char_read_all_load(buf);

    }



    while (offset < len) {

        do {

            res = s->chr_sync_read(s, buf + offset, len - offset);

            if (res == -1 && errno == EAGAIN) {

                g_usleep(100);

            }

        } while (res == -1 && errno == EAGAIN);



        if (res == 0) {

            break;

        }



        if (res < 0) {

            if (s->replay && replay_mode == REPLAY_MODE_RECORD) {

                replay_char_read_all_save_error(res);

            }

            return res;

        }



        offset += res;



        if (!counter--) {

            break;

        }

    }



    if (s->replay && replay_mode == REPLAY_MODE_RECORD) {

        replay_char_read_all_save_buf(buf, offset);

    }

    return offset;

}
