static int wsaud_read_packet(AVFormatContext *s,

                             AVPacket *pkt)

{

    AVIOContext *pb = s->pb;

    unsigned char preamble[AUD_CHUNK_PREAMBLE_SIZE];

    unsigned int chunk_size;

    int ret = 0;

    AVStream *st = s->streams[0];



    if (avio_read(pb, preamble, AUD_CHUNK_PREAMBLE_SIZE) !=

        AUD_CHUNK_PREAMBLE_SIZE)

        return AVERROR(EIO);



    /* validate the chunk */

    if (AV_RL32(&preamble[4]) != AUD_CHUNK_SIGNATURE)

        return AVERROR_INVALIDDATA;



    chunk_size = AV_RL16(&preamble[0]);



    if (st->codec->codec_id == AV_CODEC_ID_WESTWOOD_SND1) {

        /* For Westwood SND1 audio we need to add the output size and input

           size to the start of the packet to match what is in VQA.

           Specifically, this is needed to signal when a packet should be

           decoding as raw 8-bit pcm or variable-size ADPCM. */

        int out_size = AV_RL16(&preamble[2]);

        if ((ret = av_new_packet(pkt, chunk_size + 4)))

            return ret;

        if ((ret = avio_read(pb, &pkt->data[4], chunk_size)) != chunk_size)

            return ret < 0 ? ret : AVERROR(EIO);

        AV_WL16(&pkt->data[0], out_size);

        AV_WL16(&pkt->data[2], chunk_size);



        pkt->duration = out_size;

    } else {

        ret = av_get_packet(pb, pkt, chunk_size);

        if (ret != chunk_size)

            return AVERROR(EIO);



        /* 2 samples/byte, 1 or 2 samples per frame depending on stereo */

        pkt->duration = (chunk_size * 2) / st->codec->channels;

    }

    pkt->stream_index = st->index;



    return ret;

}
