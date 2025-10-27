static int mov_get_mpeg2_xdcam_codec_tag(AVFormatContext *s, MOVTrack *track)

{

    int tag = track->par->codec_tag;

    int interlaced = track->par->field_order > AV_FIELD_PROGRESSIVE;

    AVStream *st = track->st;

    int rate = av_q2d(find_fps(s, st));



    if (!tag)

        tag = MKTAG('m', '2', 'v', '1'); //fallback tag



    if (track->par->format == AV_PIX_FMT_YUV420P) {

        if (track->par->width == 1280 && track->par->height == 720) {

            if (!interlaced) {

                if      (rate == 24) tag = MKTAG('x','d','v','4');

                else if (rate == 25) tag = MKTAG('x','d','v','5');

                else if (rate == 30) tag = MKTAG('x','d','v','1');

                else if (rate == 50) tag = MKTAG('x','d','v','a');

                else if (rate == 60) tag = MKTAG('x','d','v','9');

            }

        } else if (track->par->width == 1440 && track->par->height == 1080) {

            if (!interlaced) {

                if      (rate == 24) tag = MKTAG('x','d','v','6');

                else if (rate == 25) tag = MKTAG('x','d','v','7');

                else if (rate == 30) tag = MKTAG('x','d','v','8');

            } else {

                if      (rate == 25) tag = MKTAG('x','d','v','3');

                else if (rate == 30) tag = MKTAG('x','d','v','2');

            }

        } else if (track->par->width == 1920 && track->par->height == 1080) {

            if (!interlaced) {

                if      (rate == 24) tag = MKTAG('x','d','v','d');

                else if (rate == 25) tag = MKTAG('x','d','v','e');

                else if (rate == 30) tag = MKTAG('x','d','v','f');

            } else {

                if      (rate == 25) tag = MKTAG('x','d','v','c');

                else if (rate == 30) tag = MKTAG('x','d','v','b');

            }

        }

    } else if (track->par->format == AV_PIX_FMT_YUV422P) {

        if (track->par->width == 1280 && track->par->height == 720) {

            if (!interlaced) {

                if      (rate == 24) tag = MKTAG('x','d','5','4');

                else if (rate == 25) tag = MKTAG('x','d','5','5');

                else if (rate == 30) tag = MKTAG('x','d','5','1');

                else if (rate == 50) tag = MKTAG('x','d','5','a');

                else if (rate == 60) tag = MKTAG('x','d','5','9');

            }

        } else if (track->par->width == 1920 && track->par->height == 1080) {

            if (!interlaced) {

                if      (rate == 24) tag = MKTAG('x','d','5','d');

                else if (rate == 25) tag = MKTAG('x','d','5','e');

                else if (rate == 30) tag = MKTAG('x','d','5','f');

            } else {

                if      (rate == 25) tag = MKTAG('x','d','5','c');

                else if (rate == 30) tag = MKTAG('x','d','5','b');

            }

        }

    }



    return tag;

}
