av_cold int ff_mjpeg_encode_init(MpegEncContext *s)

{

    MJpegContext *m;



    av_assert0(s->slice_context_count == 1);



    if (s->width > 65500 || s->height > 65500) {

        av_log(s, AV_LOG_ERROR, "JPEG does not support resolutions above 65500x65500\n");

        return AVERROR(EINVAL);

    }



    m = av_malloc(sizeof(MJpegContext));

    if (!m)

        return AVERROR(ENOMEM);



    s->min_qcoeff=-1023;

    s->max_qcoeff= 1023;



    // Build default Huffman tables.

    // These may be overwritten later with more optimal Huffman tables, but

    // they are needed at least right now for some processes like trellis.

    ff_mjpeg_build_huffman_codes(m->huff_size_dc_luminance,

                                 m->huff_code_dc_luminance,

                                 avpriv_mjpeg_bits_dc_luminance,

                                 avpriv_mjpeg_val_dc);

    ff_mjpeg_build_huffman_codes(m->huff_size_dc_chrominance,

                                 m->huff_code_dc_chrominance,

                                 avpriv_mjpeg_bits_dc_chrominance,

                                 avpriv_mjpeg_val_dc);

    ff_mjpeg_build_huffman_codes(m->huff_size_ac_luminance,

                                 m->huff_code_ac_luminance,

                                 avpriv_mjpeg_bits_ac_luminance,

                                 avpriv_mjpeg_val_ac_luminance);

    ff_mjpeg_build_huffman_codes(m->huff_size_ac_chrominance,

                                 m->huff_code_ac_chrominance,

                                 avpriv_mjpeg_bits_ac_chrominance,

                                 avpriv_mjpeg_val_ac_chrominance);



    ff_init_uni_ac_vlc(m->huff_size_ac_luminance,   m->uni_ac_vlc_len);

    ff_init_uni_ac_vlc(m->huff_size_ac_chrominance, m->uni_chroma_ac_vlc_len);

    s->intra_ac_vlc_length      =

    s->intra_ac_vlc_last_length = m->uni_ac_vlc_len;

    s->intra_chroma_ac_vlc_length      =

    s->intra_chroma_ac_vlc_last_length = m->uni_chroma_ac_vlc_len;



    // Buffers start out empty.

    m->huff_ncode = 0;

    s->mjpeg_ctx = m;



    return alloc_huffman(s);

}
