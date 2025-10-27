static void img_copy(uint8_t *dst, int dst_wrap, 

                     uint8_t *src, int src_wrap,

                     int width, int height)

{

    for(;height > 0; height--) {

        memcpy(dst, src, width);

        dst += dst_wrap;

        src += src_wrap;

    }

}
