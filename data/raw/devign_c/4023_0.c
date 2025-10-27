static void rgb24_to_rgb555(AVPicture *dst, AVPicture *src,

                            int width, int height)

{

    const unsigned char *p;

    unsigned char *q;

    int r, g, b, dst_wrap, src_wrap;

    int x, y;



    p = src->data[0];

    src_wrap = src->linesize[0] - 3 * width;



    q = dst->data[0];

    dst_wrap = dst->linesize[0] - 2 * width;



    for(y=0;y<height;y++) {

        for(x=0;x<width;x++) {

            r = p[0];

            g = p[1];

            b = p[2];



            ((unsigned short *)q)[0] = 

                ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3) | 0x8000;

            q += 2;

            p += 3;

        }

        p += src_wrap;

        q += dst_wrap;

    }

}
