int MP3lame_encode_frame(AVCodecContext *avctx,

                     unsigned char *frame, int buf_size, void *data)

{

	Mp3AudioContext *s = avctx->priv_data;

	int num, i;

//av_log(avctx, AV_LOG_DEBUG, "%X %d %X\n", (int)frame, buf_size, (int)data);

//        if(data==NULL)

//            return lame_encode_flush(s->gfp, frame, buf_size);



	/* lame 3.91 dies on '1-channel interleaved' data */

	if (s->stereo) {

		num = lame_encode_buffer_interleaved(s->gfp, data,

			MPA_FRAME_SIZE, frame, buf_size);

	} else {

		num = lame_encode_buffer(s->gfp, data, data, MPA_FRAME_SIZE,

			frame, buf_size);



/*av_log(avctx, AV_LOG_DEBUG, "in:%d out:%d\n", MPA_FRAME_SIZE, num);

for(i=0; i<num; i++){

    av_log(avctx, AV_LOG_DEBUG, "%2X ", frame[i]);

}*/

	}



	return num;

}
