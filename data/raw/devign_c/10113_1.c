static int fileTest(uint8_t *ref[4], int refStride[4], int w, int h, FILE *fp,

                    enum AVPixelFormat srcFormat_in,

                    enum AVPixelFormat dstFormat_in)

{

    char buf[256];



    while (fgets(buf, sizeof(buf), fp)) {

        struct Results r;

        enum AVPixelFormat srcFormat;

        char srcStr[12];

        int srcW, srcH;

        enum AVPixelFormat dstFormat;

        char dstStr[12];

        int dstW, dstH;

        int flags;

        int ret;



        ret = sscanf(buf,

                     " %12s %dx%d -> %12s %dx%d flags=%d CRC=%x"

                     " SSD=%"SCNu64 ", %"SCNu64 ", %"SCNu64 ", %"SCNu64 "\n",

                     srcStr, &srcW, &srcH, dstStr, &dstW, &dstH,

                     &flags, &r.crc, &r.ssdY, &r.ssdU, &r.ssdV, &r.ssdA);

        if (ret != 12) {

            srcStr[0] = dstStr[0] = 0;

            ret       = sscanf(buf, "%12s -> %12s\n", srcStr, dstStr);

        }



        srcFormat = av_get_pix_fmt(srcStr);

        dstFormat = av_get_pix_fmt(dstStr);



        if (srcFormat == AV_PIX_FMT_NONE || dstFormat == AV_PIX_FMT_NONE ||

            srcW > 8192U || srcH > 8192U || dstW > 8192U || dstH > 8192U) {

            fprintf(stderr, "malformed input file\n");

            return -1;

        }

        if ((srcFormat_in != AV_PIX_FMT_NONE && srcFormat_in != srcFormat) ||

            (dstFormat_in != AV_PIX_FMT_NONE && dstFormat_in != dstFormat))

            continue;

        if (ret != 12) {

            printf("%s", buf);

            continue;

        }



        doTest(ref, refStride, w, h,

               srcFormat, dstFormat,

               srcW, srcH, dstW, dstH, flags,

               &r);

    }



    return 0;

}
