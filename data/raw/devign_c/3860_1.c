static enum AVPixelFormat get_chroma_format(SchroChromaFormat schro_pix_fmt)

{

    int num_formats = sizeof(schro_pixel_format_map) /

                      sizeof(schro_pixel_format_map[0]);

    int idx;



    for (idx = 0; idx < num_formats; ++idx)

        if (schro_pixel_format_map[idx].schro_pix_fmt == schro_pix_fmt)

            return schro_pixel_format_map[idx].ff_pix_fmt;

    return AV_PIX_FMT_NONE;

}
