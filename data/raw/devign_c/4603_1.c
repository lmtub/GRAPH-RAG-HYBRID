int av_image_fill_black(uint8_t *dst_data[4], const ptrdiff_t dst_linesize[4],

                        enum AVPixelFormat pix_fmt, enum AVColorRange range,

                        int width, int height)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);

    int nb_planes = av_pix_fmt_count_planes(pix_fmt);

    // A pixel or a group of pixels on each plane, with a value that represents black.

    // Consider e.g. AV_PIX_FMT_UYVY422 for non-trivial cases.

    uint8_t clear_block[4][MAX_BLOCK_SIZE] = {0}; // clear padding with 0

    int clear_block_size[4] = {0};

    ptrdiff_t plane_line_bytes[4] = {0};

    int rgb, limited;

    int plane, c;



    if (!desc || nb_planes < 1 || nb_planes > 4 || desc->flags & AV_PIX_FMT_FLAG_HWACCEL)

        return AVERROR(EINVAL);



    rgb = !!(desc->flags & AV_PIX_FMT_FLAG_RGB);

    limited = !rgb && range != AVCOL_RANGE_JPEG;



    if (desc->flags & AV_PIX_FMT_FLAG_BITSTREAM) {

        ptrdiff_t bytewidth = av_image_get_linesize(pix_fmt, width, 0);

        uint8_t *data;

        int mono = pix_fmt == AV_PIX_FMT_MONOWHITE || pix_fmt == AV_PIX_FMT_MONOBLACK;

        int fill = pix_fmt == AV_PIX_FMT_MONOWHITE ? 0xFF : 0;

        if (nb_planes != 1 || !(rgb || mono) || bytewidth < 1)

            return AVERROR(EINVAL);



        if (!dst_data)

            return 0;



        data = dst_data[0];



        // (Bitstream + alpha will be handled incorrectly - it'll remain transparent.)

        for (;height > 0; height--) {

            memset(data, fill, bytewidth);

            data += dst_linesize[0];

        }

        return 0;

    }



    for (c = 0; c < desc->nb_components; c++) {

        const AVComponentDescriptor comp = desc->comp[c];



        // We try to operate on entire non-subsampled pixel groups (for

        // AV_PIX_FMT_UYVY422 this would mean two consecutive pixels).

        clear_block_size[comp.plane] = FFMAX(clear_block_size[comp.plane], comp.step);



        if (clear_block_size[comp.plane] > MAX_BLOCK_SIZE)

            return AVERROR(EINVAL);

    }



    // Create a byte array for clearing 1 pixel (sometimes several pixels).

    for (c = 0; c < desc->nb_components; c++) {

        const AVComponentDescriptor comp = desc->comp[c];

        // (Multiple pixels happen e.g. with AV_PIX_FMT_UYVY422.)

        int w = clear_block_size[comp.plane] / comp.step;

        uint8_t *c_data[4];

        const int c_linesize[4] = {0};

        uint16_t src_array[MAX_BLOCK_SIZE];

        uint16_t src = 0;

        int x;



        if (comp.depth > 16)

            return AVERROR(EINVAL);

        if (!rgb && comp.depth < 8)

            return AVERROR(EINVAL);

        if (w < 1)

            return AVERROR(EINVAL);



        if (c == 0 && limited) {

            src = 16 << (comp.depth - 8);

        } else if ((c == 1 || c == 2) && !rgb) {

            src = 128 << (comp.depth - 8);

        } else if (c == 3) {

            // (Assume even limited YUV uses full range alpha.)

            src = (1 << comp.depth) - 1;

        }



        for (x = 0; x < w; x++)

            src_array[x] = src;



        for (x = 0; x < 4; x++)

            c_data[x] = &clear_block[x][0];



        av_write_image_line(src_array, c_data, c_linesize, desc, 0, 0, c, w);

    }



    for (plane = 0; plane < nb_planes; plane++) {

        plane_line_bytes[plane] = av_image_get_linesize(pix_fmt, width, plane);

        if (plane_line_bytes[plane] < 0)

            return AVERROR(EINVAL);

    }



    if (!dst_data)

        return 0;



    for (plane = 0; plane < nb_planes; plane++) {

        size_t bytewidth = plane_line_bytes[plane];

        uint8_t *data = dst_data[plane];

        int chroma_div = plane == 1 || plane == 2 ? desc->log2_chroma_h : 0;

        int plane_h = ((height + ( 1 << chroma_div) - 1)) >> chroma_div;



        for (; plane_h > 0; plane_h--) {

            memset_bytes(data, bytewidth, &clear_block[plane][0], clear_block_size[plane]);

            data += dst_linesize[plane];

        }

    }



    return 0;

}
