static int ir2_decode_plane_inter(Ir2Context *ctx, int width, int height, uint8_t *dst,

                                  int pitch, const uint8_t *table)

{

    int j;

    int out = 0;

    int c;

    int t;



    if (width & 1)




    for (j = 0; j < height; j++) {

        out = 0;



        while (out < width) {

            c = ir2_get_code(&ctx->gb);

            if (c >= 0x80) { /* we have a skip */

                c   -= 0x7F;

                out += c * 2;

            } else { /* add two deltas from table */

                t        = dst[out] + (((table[c * 2] - 128)*3) >> 2);

                t        = av_clip_uint8(t);

                dst[out] = t;

                out++;

                t        = dst[out] + (((table[(c * 2) + 1] - 128)*3) >> 2);

                t        = av_clip_uint8(t);

                dst[out] = t;

                out++;

            }

        }

        dst += pitch;

    }

    return 0;

}