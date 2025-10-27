int ff_rle_encode(uint8_t *outbuf, int out_size, const uint8_t *ptr , int bpp, int w, int8_t add, uint8_t xor)

{

    int count, x;

    uint8_t *out;



    out = outbuf;





        for(x = 0; x < w; x += count) {

            /* see if we can encode the next set of pixels with RLE */

            if((count = count_pixels(ptr, w-x, bpp, 1)) > 1) {

                if(out + bpp + 1 > outbuf + out_size) return -1;

                *out++ = (count ^ xor) + add;

                memcpy(out, ptr, bpp);

                out += bpp;

            } else {

                /* fall back on uncompressed */

                count = count_pixels(ptr, w-x, bpp, 0);

                *out++ = count - 1;



                if(out + bpp*count > outbuf + out_size) return -1;

                memcpy(out, ptr, bpp * count);

                out += bpp * count;

        }



        ptr += count * bpp;

    }



    return out - outbuf;

}
