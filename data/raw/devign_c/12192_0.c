static int parse(AVCodecParserContext *ctx,

                 AVCodecContext *avctx,

                 const uint8_t **out_data, int *out_size,

                 const uint8_t *data, int size)

{

    VP9ParseContext *s = ctx->priv_data;

    int marker;



    if (size <= 0) {

        *out_size = 0;

        *out_data = data;



        return 0;

    }



    if (s->n_frames > 0) {

        *out_data = data;

        *out_size = s->size[--s->n_frames];

        parse_frame(ctx, *out_data, *out_size);



        return s->n_frames > 0 ? *out_size : size /* i.e. include idx tail */;

    }



    marker = data[size - 1];

    if ((marker & 0xe0) == 0xc0) {

        int nbytes = 1 + ((marker >> 3) & 0x3);

        int n_frames = 1 + (marker & 0x7), idx_sz = 2 + n_frames * nbytes;



        if (size >= idx_sz && data[size - idx_sz] == marker) {

            const uint8_t *idx = data + size + 1 - idx_sz;

            int first = 1;



            switch (nbytes) {

#define case_n(a, rd) \

            case a: \

                while (n_frames--) { \

                    int sz = rd; \

                    idx += a; \

                    if (sz > size) { \

                        s->n_frames = 0; \

                        av_log(avctx, AV_LOG_ERROR, \

                               "Superframe packet size too big: %d > %d\n", \

                               sz, size); \

                        return AVERROR_INVALIDDATA; \

                    } \

                    if (first) { \

                        first = 0; \

                        *out_data = data; \

                        *out_size = sz; \

                        s->n_frames = n_frames; \

                    } else { \

                        s->size[n_frames] = sz; \

                    } \

                    data += sz; \

                    size -= sz; \

                } \

                parse_frame(ctx, *out_data, *out_size); \

                return *out_size



                case_n(1, *idx);

                case_n(2, AV_RL16(idx));

                case_n(3, AV_RL24(idx));

                case_n(4, AV_RL32(idx));

            }

        }

    }



    *out_data = data;

    *out_size = size;

    parse_frame(ctx, data, size);



    return size;

}
