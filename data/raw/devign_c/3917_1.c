static int tpm_passthrough_unix_write(int fd, const uint8_t *buf, uint32_t len)

{

    int ret, remain;



    remain = len;

    while (len > 0) {

        ret = write(fd, buf, remain);

        if (ret < 0) {

            if (errno != EINTR && errno != EAGAIN) {

                return -1;

            }

        } else if (ret == 0) {

            break;

        } else {

            buf += ret;

            remain -= ret;

        }

    }

    return len - remain;

}
