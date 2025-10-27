static int rpl_read_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    RPLContext *rpl = s->priv_data;

    AVStream *vst = NULL, *ast = NULL;

    int total_audio_size;

    int error = 0;



    uint32_t i;



    int32_t audio_format, chunk_catalog_offset, number_of_chunks;

    AVRational fps;



    char line[RPL_LINE_LENGTH];



    // The header for RPL/ARMovie files is 21 lines of text

    // containing the various header fields.  The fields are always

    // in the same order, and other text besides the first

    // number usually isn't important.

    // (The spec says that there exists some significance

    // for the text in a few cases; samples needed.)

    error |= read_line(pb, line, sizeof(line));      // ARMovie

    error |= read_line(pb, line, sizeof(line));      // movie name

    av_dict_set(&s->metadata, "title"    , line, 0);

    error |= read_line(pb, line, sizeof(line));      // date/copyright

    av_dict_set(&s->metadata, "copyright", line, 0);

    error |= read_line(pb, line, sizeof(line));      // author and other

    av_dict_set(&s->metadata, "author"   , line, 0);



    // video headers

    vst = avformat_new_stream(s, NULL);

    if (!vst)

        return AVERROR(ENOMEM);

    vst->codec->codec_type      = AVMEDIA_TYPE_VIDEO;

    vst->codec->codec_tag       = read_line_and_int(pb, &error);  // video format

    vst->codec->width           = read_line_and_int(pb, &error);  // video width

    vst->codec->height          = read_line_and_int(pb, &error);  // video height

    vst->codec->bits_per_coded_sample = read_line_and_int(pb, &error);  // video bits per sample

    error |= read_line(pb, line, sizeof(line));                   // video frames per second

    fps = read_fps(line, &error);

    avpriv_set_pts_info(vst, 32, fps.den, fps.num);



    // Figure out the video codec

    switch (vst->codec->codec_tag) {

#if 0

        case 122:

            vst->codec->codec_id = AV_CODEC_ID_ESCAPE122;

            break;

#endif

        case 124:

            vst->codec->codec_id = AV_CODEC_ID_ESCAPE124;

            // The header is wrong here, at least sometimes

            vst->codec->bits_per_coded_sample = 16;

            break;

        case 130:

            vst->codec->codec_id = AV_CODEC_ID_ESCAPE130;

            break;

        default:

            avpriv_report_missing_feature(s, "Video format %i",

                                          vst->codec->codec_tag);

            vst->codec->codec_id = AV_CODEC_ID_NONE;

    }



    // Audio headers



    // ARMovie supports multiple audio tracks; I don't have any

    // samples, though. This code will ignore additional tracks.

    audio_format = read_line_and_int(pb, &error);  // audio format ID

    if (audio_format) {

        ast = avformat_new_stream(s, NULL);

        if (!ast)

            return AVERROR(ENOMEM);

        ast->codec->codec_type      = AVMEDIA_TYPE_AUDIO;

        ast->codec->codec_tag       = audio_format;

        ast->codec->sample_rate     = read_line_and_int(pb, &error);  // audio bitrate

        ast->codec->channels        = read_line_and_int(pb, &error);  // number of audio channels

        ast->codec->bits_per_coded_sample = read_line_and_int(pb, &error);  // audio bits per sample

        // At least one sample uses 0 for ADPCM, which is really 4 bits

        // per sample.

        if (ast->codec->bits_per_coded_sample == 0)

            ast->codec->bits_per_coded_sample = 4;



        ast->codec->bit_rate = ast->codec->sample_rate *

                               ast->codec->bits_per_coded_sample *

                               ast->codec->channels;



        ast->codec->codec_id = AV_CODEC_ID_NONE;

        switch (audio_format) {

            case 1:

                if (ast->codec->bits_per_coded_sample == 16) {

                    // 16-bit audio is always signed

                    ast->codec->codec_id = AV_CODEC_ID_PCM_S16LE;

                    break;

                }

                // There are some other formats listed as legal per the spec;

                // samples needed.

                break;

            case 101:

                if (ast->codec->bits_per_coded_sample == 8) {

                    // The samples with this kind of audio that I have

                    // are all unsigned.

                    ast->codec->codec_id = AV_CODEC_ID_PCM_U8;

                    break;

                } else if (ast->codec->bits_per_coded_sample == 4) {

                    ast->codec->codec_id = AV_CODEC_ID_ADPCM_IMA_EA_SEAD;

                    break;

                }

                break;

        }

        if (ast->codec->codec_id == AV_CODEC_ID_NONE)

            avpriv_request_sample(s, "Audio format %i", audio_format);

        avpriv_set_pts_info(ast, 32, 1, ast->codec->bit_rate);

    } else {

        for (i = 0; i < 3; i++)

            error |= read_line(pb, line, sizeof(line));

    }



    rpl->frames_per_chunk = read_line_and_int(pb, &error);  // video frames per chunk

    if (rpl->frames_per_chunk > 1 && vst->codec->codec_tag != 124)

        av_log(s, AV_LOG_WARNING,

               "Don't know how to split frames for video format %i. "

               "Video stream will be broken!\n", vst->codec->codec_tag);



    number_of_chunks = read_line_and_int(pb, &error);  // number of chunks in the file

    // The number in the header is actually the index of the last chunk.

    number_of_chunks++;



    error |= read_line(pb, line, sizeof(line));  // "even" chunk size in bytes

    error |= read_line(pb, line, sizeof(line));  // "odd" chunk size in bytes

    chunk_catalog_offset =                       // offset of the "chunk catalog"

        read_line_and_int(pb, &error);           //   (file index)

    error |= read_line(pb, line, sizeof(line));  // offset to "helpful" sprite

    error |= read_line(pb, line, sizeof(line));  // size of "helpful" sprite

    error |= read_line(pb, line, sizeof(line));  // offset to key frame list



    // Read the index

    avio_seek(pb, chunk_catalog_offset, SEEK_SET);

    total_audio_size = 0;

    for (i = 0; !error && i < number_of_chunks; i++) {

        int64_t offset, video_size, audio_size;

        error |= read_line(pb, line, sizeof(line));

        if (3 != sscanf(line, "%"SCNd64" , %"SCNd64" ; %"SCNd64,

                        &offset, &video_size, &audio_size))

            error = -1;

        av_add_index_entry(vst, offset, i * rpl->frames_per_chunk,

                           video_size, rpl->frames_per_chunk, 0);

        if (ast)

            av_add_index_entry(ast, offset + video_size, total_audio_size,

                               audio_size, audio_size * 8, 0);

        total_audio_size += audio_size * 8;

    }



    if (error) return AVERROR(EIO);



    return 0;

}
