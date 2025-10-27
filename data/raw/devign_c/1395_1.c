int ff_h264_decode_extradata(H264Context *h)

{

    AVCodecContext *avctx = h->s.avctx;



    if (avctx->extradata[0] == 1) {

        int i, cnt, nalsize;

        unsigned char *p = avctx->extradata;



        h->is_avc = 1;



        if (avctx->extradata_size < 7) {

            av_log(avctx, AV_LOG_ERROR, "avcC too short\n");

            return -1;

        }

        /* sps and pps in the avcC always have length coded with 2 bytes,

         * so put a fake nal_length_size = 2 while parsing them */

        h->nal_length_size = 2;

        // Decode sps from avcC

        cnt = *(p + 5) & 0x1f; // Number of sps

        p  += 6;

        for (i = 0; i < cnt; i++) {

            nalsize = AV_RB16(p) + 2;

            if (p - avctx->extradata + nalsize > avctx->extradata_size)

                return -1;

            if (decode_nal_units(h, p, nalsize) < 0) {

                av_log(avctx, AV_LOG_ERROR,

                       "Decoding sps %d from avcC failed\n", i);

                return -1;

            }

            p += nalsize;

        }

        // Decode pps from avcC

        cnt = *(p++); // Number of pps

        for (i = 0; i < cnt; i++) {

            nalsize = AV_RB16(p) + 2;

            if (p - avctx->extradata + nalsize > avctx->extradata_size)

                return -1;

            if (decode_nal_units(h, p, nalsize) < 0) {

                av_log(avctx, AV_LOG_ERROR,

                       "Decoding pps %d from avcC failed\n", i);

                return -1;

            }

            p += nalsize;

        }

        // Now store right nal length size, that will be used to parse all other nals

        h->nal_length_size = (avctx->extradata[4] & 0x03) + 1;

    } else {

        h->is_avc = 0;

        if (decode_nal_units(h, avctx->extradata, avctx->extradata_size) < 0)

            return -1;

    }

    return 0;

}
