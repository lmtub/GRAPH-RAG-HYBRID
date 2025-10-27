static void print_codec(const AVCodec *c)

{

    int encoder = av_codec_is_encoder(c);



    printf("%s %s [%s]:\n", encoder ? "Encoder" : "Decoder", c->name,

           c->long_name ? c->long_name : "");



    printf("    General capabilities: ");

    if (c->capabilities & AV_CODEC_CAP_DRAW_HORIZ_BAND)

        printf("horizband ");

    if (c->capabilities & AV_CODEC_CAP_DR1)

        printf("dr1 ");

    if (c->capabilities & AV_CODEC_CAP_TRUNCATED)

        printf("trunc ");

    if (c->capabilities & AV_CODEC_CAP_DELAY)

        printf("delay ");

    if (c->capabilities & AV_CODEC_CAP_SMALL_LAST_FRAME)

        printf("small ");

    if (c->capabilities & AV_CODEC_CAP_SUBFRAMES)

        printf("subframes ");

    if (c->capabilities & AV_CODEC_CAP_EXPERIMENTAL)

        printf("exp ");

    if (c->capabilities & AV_CODEC_CAP_CHANNEL_CONF)

        printf("chconf ");

    if (c->capabilities & AV_CODEC_CAP_PARAM_CHANGE)

        printf("small ");

    if (c->capabilities & AV_CODEC_CAP_PARAM_CHANGE)

        printf("variable ");

    if (c->capabilities & (AV_CODEC_CAP_FRAME_THREADS |

                           AV_CODEC_CAP_SLICE_THREADS |

                           AV_CODEC_CAP_AUTO_THREADS))

        printf("threads ");

    if (!c->capabilities)

        printf("none");

    printf("\n");



    if (c->type == AVMEDIA_TYPE_VIDEO) {

        printf("    Threading capabilities: ");

        switch (c->capabilities & (AV_CODEC_CAP_FRAME_THREADS |

                                   AV_CODEC_CAP_SLICE_THREADS |

                                   AV_CODEC_CAP_AUTO_THREADS)) {

        case AV_CODEC_CAP_FRAME_THREADS |

             AV_CODEC_CAP_SLICE_THREADS: printf("frame and slice"); break;

        case AV_CODEC_CAP_FRAME_THREADS: printf("frame");           break;

        case AV_CODEC_CAP_SLICE_THREADS: printf("slice");           break;

        case AV_CODEC_CAP_AUTO_THREADS : printf("auto");            break;

        default:                         printf("none");            break;

        }

        printf("\n");

    }



    if (c->supported_framerates) {

        const AVRational *fps = c->supported_framerates;



        printf("    Supported framerates:");

        while (fps->num) {

            printf(" %d/%d", fps->num, fps->den);

            fps++;

        }

        printf("\n");

    }

    PRINT_CODEC_SUPPORTED(c, pix_fmts, enum AVPixelFormat, "pixel formats",

                          AV_PIX_FMT_NONE, GET_PIX_FMT_NAME);

    PRINT_CODEC_SUPPORTED(c, supported_samplerates, int, "sample rates", 0,

                          GET_SAMPLE_RATE_NAME);

    PRINT_CODEC_SUPPORTED(c, sample_fmts, enum AVSampleFormat, "sample formats",

                          AV_SAMPLE_FMT_NONE, GET_SAMPLE_FMT_NAME);

    PRINT_CODEC_SUPPORTED(c, channel_layouts, uint64_t, "channel layouts",

                          0, GET_CH_LAYOUT_DESC);



    if (c->priv_class) {

        show_help_children(c->priv_class,

                           AV_OPT_FLAG_ENCODING_PARAM |

                           AV_OPT_FLAG_DECODING_PARAM);

    }

}
