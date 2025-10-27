static int encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                        const AVFrame *pict, int *got_packet)

{

    TiffEncoderContext *s = avctx->priv_data;

    const AVFrame *const p = pict;

    int i;

    uint8_t *ptr;

    uint8_t *offset;

    uint32_t strips;

    uint32_t *strip_sizes   = NULL;

    uint32_t *strip_offsets = NULL;

    int bytes_per_row;

    uint32_t res[2]    = { 72, 1 };     // image resolution (72/1)

    uint16_t bpp_tab[] = { 8, 8, 8, 8 };

    int ret;

    int is_yuv = 0;

    uint8_t *yuv_line = NULL;

    int shift_h, shift_v;

    int packet_size;

    const AVPixFmtDescriptor *pfd;



    s->avctx = avctx;



    s->width          = avctx->width;

    s->height         = avctx->height;

    s->subsampling[0] = 1;

    s->subsampling[1] = 1;



    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_RGB48LE:

    case AV_PIX_FMT_GRAY16LE:

    case AV_PIX_FMT_RGBA:

    case AV_PIX_FMT_RGB24:

    case AV_PIX_FMT_GRAY8:

    case AV_PIX_FMT_PAL8:

        pfd    = av_pix_fmt_desc_get(avctx->pix_fmt);

        s->bpp = av_get_bits_per_pixel(pfd);

        if (pfd->flags & AV_PIX_FMT_FLAG_PAL)

            s->photometric_interpretation = TIFF_PHOTOMETRIC_PALETTE;

        else if (pfd->flags & AV_PIX_FMT_FLAG_RGB)

            s->photometric_interpretation = TIFF_PHOTOMETRIC_RGB;

        else

            s->photometric_interpretation = TIFF_PHOTOMETRIC_BLACK_IS_ZERO;

        s->bpp_tab_size = pfd->nb_components;

        for (i = 0; i < s->bpp_tab_size; i++)

            bpp_tab[i] = s->bpp / s->bpp_tab_size;

        break;

    case AV_PIX_FMT_MONOBLACK:

        s->bpp                        = 1;

        s->photometric_interpretation = TIFF_PHOTOMETRIC_BLACK_IS_ZERO;

        s->bpp_tab_size               = 0;

        break;

    case AV_PIX_FMT_MONOWHITE:

        s->bpp                        = 1;

        s->photometric_interpretation = TIFF_PHOTOMETRIC_WHITE_IS_ZERO;

        s->bpp_tab_size               = 0;

        break;

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUV422P:

    case AV_PIX_FMT_YUV444P:

    case AV_PIX_FMT_YUV410P:

    case AV_PIX_FMT_YUV411P:

        av_pix_fmt_get_chroma_sub_sample(avctx->pix_fmt, &shift_h, &shift_v);

        s->photometric_interpretation = TIFF_PHOTOMETRIC_YCBCR;

        s->bpp                        = 8 + (16 >> (shift_h + shift_v));

        s->subsampling[0]             = 1 << shift_h;

        s->subsampling[1]             = 1 << shift_v;

        s->bpp_tab_size               = 3;

        is_yuv                        = 1;

        break;

    default:

        av_log(s->avctx, AV_LOG_ERROR,

               "This colors format is not supported\n");

        return -1;

    }



    if (s->compr == TIFF_DEFLATE       ||

        s->compr == TIFF_ADOBE_DEFLATE ||

        s->compr == TIFF_LZW)

        // best choice for DEFLATE

        s->rps = s->height;

    else

        // suggest size of strip

        s->rps = FFMAX(8192 / (((s->width * s->bpp) >> 3) + 1), 1);

    // round rps up

    s->rps = ((s->rps - 1) / s->subsampling[1] + 1) * s->subsampling[1];



    strips = (s->height - 1) / s->rps + 1;



    packet_size = avctx->height * ((avctx->width * s->bpp + 7) >> 3) * 2 +

                  avctx->height * 4 + FF_MIN_BUFFER_SIZE;



    if (!pkt->data &&

        (ret = av_new_packet(pkt, packet_size)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet.\n");

        return ret;

    }

    ptr          = pkt->data;

    s->buf_start = pkt->data;

    s->buf       = &ptr;

    s->buf_size  = pkt->size;



    if (check_size(s, 8))

        goto fail;



    // write header

    bytestream_put_le16(&ptr, 0x4949);

    bytestream_put_le16(&ptr, 42);



    offset = ptr;

    bytestream_put_le32(&ptr, 0);



    strip_sizes   = av_mallocz_array(strips, sizeof(*strip_sizes));

    strip_offsets = av_mallocz_array(strips, sizeof(*strip_offsets));

    if (!strip_sizes || !strip_offsets) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    bytes_per_row = (((s->width - 1) / s->subsampling[0] + 1) * s->bpp *

                     s->subsampling[0] * s->subsampling[1] + 7) >> 3;

    if (is_yuv) {

        yuv_line = av_malloc(bytes_per_row);

        if (!yuv_line) {

            av_log(s->avctx, AV_LOG_ERROR, "Not enough memory\n");

            ret = AVERROR(ENOMEM);

            goto fail;

        }

    }



#if CONFIG_ZLIB

    if (s->compr == TIFF_DEFLATE || s->compr == TIFF_ADOBE_DEFLATE) {

        uint8_t *zbuf;

        int zlen, zn;

        int j;



        zlen = bytes_per_row * s->rps;

        zbuf = av_malloc(zlen);

        if (!zbuf) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        strip_offsets[0] = ptr - pkt->data;

        zn               = 0;

        for (j = 0; j < s->rps; j++) {

            if (is_yuv) {

                pack_yuv(s, p, yuv_line, j);

                memcpy(zbuf + zn, yuv_line, bytes_per_row);

                j += s->subsampling[1] - 1;

            } else

                memcpy(zbuf + j * bytes_per_row,

                       p->data[0] + j * p->linesize[0], bytes_per_row);

            zn += bytes_per_row;

        }

        ret = encode_strip(s, zbuf, ptr, zn, s->compr);

        av_free(zbuf);

        if (ret < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "Encode strip failed\n");

            goto fail;

        }

        ptr           += ret;

        strip_sizes[0] = ptr - pkt->data - strip_offsets[0];

    } else

#endif

    if (s->compr == TIFF_LZW) {

        s->lzws = av_malloc(ff_lzw_encode_state_size);

        if (!s->lzws) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

    }

    for (i = 0; i < s->height; i++) {

        if (strip_sizes[i / s->rps] == 0) {

            if (s->compr == TIFF_LZW) {

                ff_lzw_encode_init(s->lzws, ptr,

                                   s->buf_size - (*s->buf - s->buf_start),

                                   12, FF_LZW_TIFF, put_bits);

            }

            strip_offsets[i / s->rps] = ptr - pkt->data;

        }

        if (is_yuv) {

            pack_yuv(s, p, yuv_line, i);

            ret = encode_strip(s, yuv_line, ptr, bytes_per_row, s->compr);

            i  += s->subsampling[1] - 1;

        } else

            ret = encode_strip(s, p->data[0] + i * p->linesize[0],

                               ptr, bytes_per_row, s->compr);

        if (ret < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "Encode strip failed\n");

            goto fail;

        }

        strip_sizes[i / s->rps] += ret;

        ptr                     += ret;

        if (s->compr == TIFF_LZW &&

            (i == s->height - 1 || i % s->rps == s->rps - 1)) {

            ret = ff_lzw_encode_flush(s->lzws, flush_put_bits);

            strip_sizes[(i / s->rps)] += ret;

            ptr                       += ret;

        }

    }

    if (s->compr == TIFF_LZW)

        av_free(s->lzws);



    s->num_entries = 0;



    add_entry1(s, TIFF_SUBFILE, TIFF_LONG, 0);

    add_entry1(s, TIFF_WIDTH,   TIFF_LONG, s->width);

    add_entry1(s, TIFF_HEIGHT,  TIFF_LONG, s->height);



    if (s->bpp_tab_size)

        add_entry(s, TIFF_BPP, TIFF_SHORT, s->bpp_tab_size, bpp_tab);



    add_entry1(s, TIFF_COMPR,       TIFF_SHORT, s->compr);

    add_entry1(s, TIFF_PHOTOMETRIC, TIFF_SHORT, s->photometric_interpretation);

    add_entry(s,  TIFF_STRIP_OFFS,  TIFF_LONG,  strips, strip_offsets);



    if (s->bpp_tab_size)

        add_entry1(s, TIFF_SAMPLES_PER_PIXEL, TIFF_SHORT, s->bpp_tab_size);



    add_entry1(s, TIFF_ROWSPERSTRIP, TIFF_LONG,     s->rps);

    add_entry(s,  TIFF_STRIP_SIZE,   TIFF_LONG,     strips, strip_sizes);

    add_entry(s,  TIFF_XRES,         TIFF_RATIONAL, 1,      res);

    add_entry(s,  TIFF_YRES,         TIFF_RATIONAL, 1,      res);

    add_entry1(s, TIFF_RES_UNIT,     TIFF_SHORT,    2);



    if (!(avctx->flags & CODEC_FLAG_BITEXACT))

        add_entry(s, TIFF_SOFTWARE_NAME, TIFF_STRING,

                  strlen(LIBAVCODEC_IDENT) + 1, LIBAVCODEC_IDENT);



    if (avctx->pix_fmt == AV_PIX_FMT_PAL8) {

        uint16_t pal[256 * 3];

        for (i = 0; i < 256; i++) {

            uint32_t rgb = *(uint32_t *) (p->data[1] + i * 4);

            pal[i]       = ((rgb >> 16) & 0xff) * 257;

            pal[i + 256] = ((rgb >>  8) & 0xff) * 257;

            pal[i + 512] =  (rgb        & 0xff) * 257;

        }

        add_entry(s, TIFF_PAL, TIFF_SHORT, 256 * 3, pal);

    }

    if (is_yuv) {

        /** according to CCIR Recommendation 601.1 */

        uint32_t refbw[12] = { 15, 1, 235, 1, 128, 1, 240, 1, 128, 1, 240, 1 };

        add_entry(s, TIFF_YCBCR_SUBSAMPLING, TIFF_SHORT,    2, s->subsampling);

        add_entry(s, TIFF_REFERENCE_BW,      TIFF_RATIONAL, 6, refbw);

    }

    // write offset to dir

    bytestream_put_le32(&offset, ptr - pkt->data);



    if (check_size(s, 6 + s->num_entries * 12)) {

        ret = AVERROR(EINVAL);

        goto fail;

    }

    bytestream_put_le16(&ptr, s->num_entries);  // write tag count

    bytestream_put_buffer(&ptr, s->entries, s->num_entries * 12);

    bytestream_put_le32(&ptr, 0);



    pkt->size   = ptr - pkt->data;

    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



fail:

    av_free(strip_sizes);

    av_free(strip_offsets);

    av_free(yuv_line);

    return ret;

}
