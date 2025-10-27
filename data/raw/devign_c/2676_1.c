static int read_uncompressed_sgi(unsigned char* out_buf, uint8_t* out_end,

                const uint8_t *in_buf, const uint8_t *in_end, SgiState* s)

{

    int x, y, z;

    const uint8_t *ptr;

    unsigned int offset = s->height * s->width * s->bytes_per_channel;



    /* Test buffer size. */

    if (offset * s->depth > in_end - in_buf) {

       return -1;

    }



    for (y = s->height - 1; y >= 0; y--) {

        out_end = out_buf + (y * s->linesize);

        for (x = s->width; x > 0; x--) {

            ptr = in_buf += s->bytes_per_channel;

            for(z = 0; z < s->depth; z ++) {

                memcpy(out_end, ptr, s->bytes_per_channel);

                out_end += s->bytes_per_channel;

                ptr += offset;

            }

        }

    }

    return 0;

}
