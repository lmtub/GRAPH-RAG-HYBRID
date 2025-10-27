static void to_meta_with_crop(AVCodecContext *avctx, AVFrame *p, int *dest)

{

    int blockx, blocky, x, y;

    int luma = 0;

    int height = FFMIN(avctx->height, C64YRES);

    int width  = FFMIN(avctx->width , C64XRES);

    uint8_t *src = p->data[0];



    for (blocky = 0; blocky < C64YRES; blocky += 8) {

        for (blockx = 0; blockx < C64XRES; blockx += 8) {

            for (y = blocky; y < blocky + 8 && y < C64YRES; y++) {

                for (x = blockx; x < blockx + 8 && x < C64XRES; x += 2) {

                    if(x < width && y < height) {

                        /* build average over 2 pixels */

                        luma = (src[(x + 0 + y * p->linesize[0])] +

                                src[(x + 1 + y * p->linesize[0])]) / 2;

                        /* write blocks as linear data now so they are suitable for elbg */

                        dest[0] = luma;

                    }

                    dest++;

                }

            }

        }

    }

}
