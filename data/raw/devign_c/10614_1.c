static int caf_write_trailer(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    AVCodecContext *enc = s->streams[0]->codec;



    if (pb->seekable) {

        CAFContext *caf = s->priv_data;

        int64_t file_size = avio_tell(pb);



        avio_seek(pb, caf->data, SEEK_SET);

        avio_wb64(pb, file_size - caf->data - 8);

        avio_seek(pb, file_size, SEEK_SET);

        if (!enc->block_align) {

            ffio_wfourcc(pb, "pakt");

            avio_wb64(pb, caf->size_entries_used + 24);

            avio_wb64(pb, caf->packets); ///< mNumberPackets

            avio_wb64(pb, caf->packets * samples_per_packet(enc->codec_id, enc->channels)); ///< mNumberValidFrames

            avio_wb32(pb, 0); ///< mPrimingFrames

            avio_wb32(pb, 0); ///< mRemainderFrames

            avio_write(pb, caf->pkt_sizes, caf->size_entries_used);

            av_freep(&caf->pkt_sizes);

            caf->size_buffer_size = 0;

        }

        avio_flush(pb);

    }

    return 0;

}
