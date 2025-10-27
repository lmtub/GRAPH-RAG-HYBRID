void checkasm_check_h264pred(void)

{

    static const struct {

        void (*func)(H264PredContext*, uint8_t*, uint8_t*, int, int, int);

        const char *name;

    } tests[] = {

        { check_pred4x4,   "pred4x4"   },

        { check_pred8x8,   "pred8x8"   },

        { check_pred16x16, "pred16x16" },

        { check_pred8x8l,  "pred8x8l"  },

    };



    DECLARE_ALIGNED(16, uint8_t, buf0)[BUF_SIZE];

    DECLARE_ALIGNED(16, uint8_t, buf1)[BUF_SIZE];

    H264PredContext h;

    int test, codec, chroma_format, bit_depth;



    for (test = 0; test < FF_ARRAY_ELEMS(tests); test++) {

        for (codec = 0; codec < 4; codec++) {

            int codec_id = codec_ids[codec];

            for (bit_depth = 8; bit_depth <= (codec_id == AV_CODEC_ID_H264 ? 10 : 8); bit_depth++)

                for (chroma_format = 1; chroma_format <= (codec_id == AV_CODEC_ID_H264 ? 2 : 1); chroma_format++) {

                    ff_h264_pred_init(&h, codec_id, bit_depth, chroma_format);

                    tests[test].func(&h, buf0, buf1, codec, chroma_format, bit_depth);

                }

        }

        report("%s", tests[test].name);

    }

}
