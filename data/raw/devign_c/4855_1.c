static int xbm_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame, AVPacket *avpkt)

{

    AVFrame *p = data;

    const uint8_t *end, *ptr = avpkt->data;

    uint8_t *dst;

    int ret, linesize, i, j;



    end = avpkt->data + avpkt->size;

    while (!avctx->width || !avctx->height) {

        char name[256];

        int number, len;



        ptr += strcspn(ptr, "#");

        if (sscanf(ptr, "#define %256s %u", name, &number) != 2) {

            av_log(avctx, AV_LOG_ERROR, "Unexpected preprocessor directive\n");

            return AVERROR_INVALIDDATA;

        }



        len = strlen(name);

        if ((len > 6) && !avctx->height && !memcmp(name + len - 7, "_height", 7)) {

                avctx->height = number;

        } else if ((len > 5) && !avctx->width && !memcmp(name + len - 6, "_width", 6)) {

                avctx->width = number;

        } else {

            av_log(avctx, AV_LOG_ERROR, "Unknown define '%s'\n", name);

            return AVERROR_INVALIDDATA;

        }

        ptr += strcspn(ptr, "\n\r") + 1;

    }



    if ((ret = ff_get_buffer(avctx, p, 0)) < 0)

        return ret;



    // goto start of image data

    ptr += strcspn(ptr, "{") + 1;



    linesize = (avctx->width + 7) / 8;

    for (i = 0; i < avctx->height; i++) {

        dst = p->data[0] + i * p->linesize[0];

        for (j = 0; j < linesize; j++) {

            uint8_t val;



            ptr += strcspn(ptr, "x") + 1;

            if (ptr < end && av_isxdigit(*ptr)) {

                val = convert(*ptr);

                ptr++;

                if (av_isxdigit(*ptr))

                    val = (val << 4) + convert(*ptr);

                *dst++ = ff_reverse[val];

            } else {

                av_log(avctx, AV_LOG_ERROR, "Unexpected data at '%.8s'\n", ptr);

                return AVERROR_INVALIDDATA;

            }

        }

    }



    p->key_frame = 1;

    p->pict_type = AV_PICTURE_TYPE_I;



    *got_frame       = 1;



    return avpkt->size;

}
