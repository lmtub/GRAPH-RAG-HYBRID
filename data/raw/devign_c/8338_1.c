static int ebml_read_binary(AVIOContext *pb, int length, EbmlBin *bin)

{

    av_free(bin->data);

    if (!(bin->data = av_malloc(length)))

        return AVERROR(ENOMEM);



    bin->size = length;

    bin->pos  = avio_tell(pb);

    if (avio_read(pb, bin->data, length) != length) {

        av_freep(&bin->data);

        return AVERROR(EIO);

    }



    return 0;

}
