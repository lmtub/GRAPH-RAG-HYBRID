int av_find_best_stream(AVFormatContext *ic,

                        enum AVMediaType type,

                        int wanted_stream_nb,

                        int related_stream,

                        AVCodec **decoder_ret,

                        int flags)

{

    int i, nb_streams = ic->nb_streams, stream_number = 0;

    int ret = AVERROR_STREAM_NOT_FOUND, best_count = -1;

    unsigned *program = NULL;

    AVCodec *decoder = NULL, *best_decoder = NULL;



    if (related_stream >= 0 && wanted_stream_nb < 0) {

        AVProgram *p = find_program_from_stream(ic, related_stream);

        if (p) {

            program = p->stream_index;

            nb_streams = p->nb_stream_indexes;

        }

    }

    for (i = 0; i < nb_streams; i++) {

        AVStream *st = ic->streams[program ? program[i] : i];

        AVCodecContext *avctx = st->codec;

        if (avctx->codec_type != type)

            continue;

        if (wanted_stream_nb >= 0 && stream_number++ != wanted_stream_nb)

            continue;

        if (st->disposition & (AV_DISPOSITION_HEARING_IMPAIRED|AV_DISPOSITION_VISUAL_IMPAIRED))

            continue;

        if (decoder_ret) {

            decoder = avcodec_find_decoder(st->codec->codec_id);

            if (!decoder) {

                if (ret < 0)

                    ret = AVERROR_DECODER_NOT_FOUND;

                continue;

            }

        }

        if (best_count >= st->codec_info_nb_frames)

            continue;

        best_count = st->codec_info_nb_frames;

        ret = program ? program[i] : i;

        best_decoder = decoder;

        if (program && i == nb_streams - 1 && ret < 0) {

            program = NULL;

            nb_streams = ic->nb_streams;

            i = 0; /* no related stream found, try again with everything */

        }

    }

    if (decoder_ret)

        *decoder_ret = best_decoder;

    return ret;

}
