static int find_marker(const uint8_t **pbuf_ptr, const uint8_t *buf_end)

{

    const uint8_t *buf_ptr;

    unsigned int v, v2;

    int val;

    int skipped = 0;



    buf_ptr = *pbuf_ptr;

    while (buf_ptr < buf_end) {

        v  = *buf_ptr++;

        v2 = *buf_ptr;

        if ((v == 0xff) && (v2 >= 0xc0) && (v2 <= 0xfe) && buf_ptr < buf_end) {

            val = *buf_ptr++;

            goto found;

        }

        skipped++;

    }

    val = -1;

found:

    av_dlog(NULL, "find_marker skipped %d bytes\n", skipped);

    *pbuf_ptr = buf_ptr;

    return val;

}
