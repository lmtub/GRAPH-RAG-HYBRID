int avcodec_decode_video(AVCodecContext *avctx, AVPicture *picture, 

                         int *got_picture_ptr,

                         UINT8 *buf, int buf_size)

{

    int ret;



    ret = avctx->codec->decode(avctx, picture, got_picture_ptr, 

                               buf, buf_size);

    avctx->frame_number++;

    return ret;

}
