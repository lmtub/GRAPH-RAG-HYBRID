static PayloadContext *h264_new_extradata(void)

{

    PayloadContext *data =

        av_mallocz(sizeof(PayloadContext) +

                   FF_INPUT_BUFFER_PADDING_SIZE);



    if (data) {

        data->cookie = MAGIC_COOKIE;

    }



    return data;

}
