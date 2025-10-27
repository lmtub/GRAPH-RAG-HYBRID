static int http_read(URLContext *h, uint8_t *buf, int size)

{

    HTTPContext *s = h->priv_data;

    int size1, len;



    size1 = size;

    while (size > 0) {

        /* read bytes from input buffer first */

        len = s->buf_end - s->buf_ptr;

        if (len > 0) {

            if (len > size)

                len = size;

            memcpy(buf, s->buf_ptr, len);

            s->buf_ptr += len;

        } else {

            len = url_read (s->hd, buf, size);

            if (len < 0) {

                return len;

            } else if (len == 0) {

                break;

            }

        }

        size -= len;

        buf += len;

    }

    return size1 - size;

}
