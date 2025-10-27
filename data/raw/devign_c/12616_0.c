static int encode_hq_slice(AVCodecContext *avctx, void *arg)

{

    SliceArgs *slice_dat = arg;

    VC2EncContext *s = slice_dat->ctx;

    PutBitContext *pb = &slice_dat->pb;

    const int slice_x = slice_dat->x;

    const int slice_y = slice_dat->y;

    const int quant_idx = slice_dat->quant_idx;

    const int slice_bytes_max = slice_dat->bytes;

    uint8_t quants[MAX_DWT_LEVELS][4];

    int p, level, orientation;



    avpriv_align_put_bits(pb);

    skip_put_bytes(pb, s->prefix_bytes);

    put_bits(pb, 8, quant_idx);



    /* Slice quantization (slice_quantizers() in the specs) */

    for (level = 0; level < s->wavelet_depth; level++)

        for (orientation = !!level; orientation < 4; orientation++)

            quants[level][orientation] = FFMAX(quant_idx - s->quant[level][orientation], 0);



    /* Luma + 2 Chroma planes */

    for (p = 0; p < 3; p++) {

        int bytes_start, bytes_len, pad_s, pad_c;

        bytes_start = put_bits_count(pb) >> 3;

        put_bits(pb, 8, 0);

        for (level = 0; level < s->wavelet_depth; level++) {

            for (orientation = !!level; orientation < 4; orientation++) {

                encode_subband(s, pb, slice_x, slice_y,

                               &s->plane[p].band[level][orientation],

                               quants[level][orientation]);

            }

        }

        avpriv_align_put_bits(pb);

        bytes_len = (put_bits_count(pb) >> 3) - bytes_start - 1;

        if (p == 2) {

            int len_diff = slice_bytes_max - (put_bits_count(pb) >> 3);

            pad_s = FFALIGN((bytes_len + len_diff), s->size_scaler)/s->size_scaler;

            pad_c = (pad_s*s->size_scaler) - bytes_len;

        } else {

            pad_s = FFALIGN(bytes_len, s->size_scaler)/s->size_scaler;

            pad_c = (pad_s*s->size_scaler) - bytes_len;

        }

        pb->buf[bytes_start] = pad_s;

        flush_put_bits(pb);

        skip_put_bytes(pb, pad_c);

    }



    return 0;

}
