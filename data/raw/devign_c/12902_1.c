void av_image_copy_plane(uint8_t       *dst, int dst_linesize,

                         const uint8_t *src, int src_linesize,

                         int bytewidth, int height)

{

    if (!dst || !src)

        return;



    for (;height > 0; height--) {

        memcpy(dst, src, bytewidth);

        dst += dst_linesize;

        src += src_linesize;

    }

}