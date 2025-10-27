static int ast_write_trailer(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    ASTMuxContext *ast = s->priv_data;

    AVCodecContext *enc = s->streams[0]->codec;

    int64_t file_size = avio_tell(pb);

    int64_t samples = (file_size - 64 - (32 * enc->frame_number)) / enc->block_align; /* PCM_S16BE_PLANAR */



    av_log(s, AV_LOG_DEBUG, "total samples: %"PRId64"\n", samples);



    if (s->pb->seekable) {

        /* Number of samples */

        avio_seek(pb, ast->samples, SEEK_SET);

        avio_wb32(pb, samples);



        /* Loopstart if provided */

        if (ast->loopstart > 0) {

        if (ast->loopstart >= samples) {

            av_log(s, AV_LOG_WARNING, "Loopstart value is out of range and will be ignored\n");

            ast->loopstart = -1;

            avio_skip(pb, 4);

        } else

        avio_wb32(pb, ast->loopstart);

        } else

            avio_skip(pb, 4);



        /* Loopend if provided. Otherwise number of samples again */

        if (ast->loopend && ast->loopstart >= 0) {

            if (ast->loopend > samples) {

                av_log(s, AV_LOG_WARNING, "Loopend value is out of range and will be ignored\n");

                ast->loopend = samples;

            }

            avio_wb32(pb, ast->loopend);

        } else {

            avio_wb32(pb, samples);

        }



        /* Size of first block */

        avio_wb32(pb, ast->fbs);



        /* File size minus header */

        avio_seek(pb, ast->size, SEEK_SET);

        avio_wb32(pb, file_size - 64);



        /* Loop flag */

        if (ast->loopstart >= 0) {

            avio_skip(pb, 6);

            avio_wb16(pb, 0xFFFF);

        }



        avio_seek(pb, file_size, SEEK_SET);

        avio_flush(pb);

    }

    return 0;

}
