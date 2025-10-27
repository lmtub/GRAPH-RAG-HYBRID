static int adx_encode_frame(AVCodecContext *avctx,

			    unsigned char *frame, int buf_size, const void *data)

{

	ADXContext *c = avctx->priv_data;

	const short *samples = data;

	unsigned char *dst = frame;

	int rest = avctx->frame_size;



/*

	input data size =

	ffmpeg.c: do_audio_out()

	frame_bytes = enc->frame_size * 2 * enc->channels;

*/



//	printf("sz=%d ",buf_size); fflush(stdout);

	if (!c->header_parsed) {

		int hdrsize = adx_encode_header(avctx,dst,buf_size);

		dst+=hdrsize;

		c->header_parsed = 1;

	}



	if (avctx->channels==1) {

		while(rest>=32) {

			adx_encode(dst,samples,c->prev);

			dst+=18;

			samples+=32;

			rest-=32;

		}

	} else {

		while(rest>=32*2) {

			short tmpbuf[32*2];

			int i;



			for(i=0;i<32;i++) {

				tmpbuf[i] = samples[i*2];

				tmpbuf[i+32] = samples[i*2+1];

			}



			adx_encode(dst,tmpbuf,c->prev);

			adx_encode(dst+18,tmpbuf+32,c->prev+1);

			dst+=18*2;

			samples+=32*2;

			rest-=32*2;

		}

	}

	return dst-frame;

}
