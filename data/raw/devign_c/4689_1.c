AVFrameSideData *av_frame_new_side_data(AVFrame *frame,

                                        enum AVFrameSideDataType type,

                                        int size)

{

    AVFrameSideData *ret, **tmp;



    if (frame->nb_side_data > INT_MAX / sizeof(*frame->side_data) - 1)

        return NULL;



    tmp = av_realloc(frame->side_data,

                     (frame->nb_side_data + 1) * sizeof(*frame->side_data));

    if (!tmp)

        return NULL;

    frame->side_data = tmp;



    ret = av_mallocz(sizeof(*ret));

    if (!ret)

        return NULL;



    if (size > 0) {

        ret->buf = av_buffer_alloc(size);

        if (!ret->buf) {

            av_freep(&ret);

            return NULL;

        }



        ret->data = ret->buf->data;

        ret->size = size;

    }

    ret->type = type;



    frame->side_data[frame->nb_side_data++] = ret;



    return ret;

}
