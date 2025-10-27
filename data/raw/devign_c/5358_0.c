static int svq1_decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             UINT8 *buf, int buf_size)

{

  MpegEncContext *s=avctx->priv_data;

  uint8_t      *current, *previous;

  int		result, i, x, y, width, height;

  AVFrame *pict = data; 



  /* initialize bit buffer */

  init_get_bits(&s->gb,buf,buf_size);



  /* decode frame header */

  s->f_code = get_bits (&s->gb, 22);



  if ((s->f_code & ~0x70) || !(s->f_code & 0x60))

    return -1;



  /* swap some header bytes (why?) */

  if (s->f_code != 0x20) {

    uint32_t *src = (uint32_t *) (buf + 4);



    for (i=0; i < 4; i++) {

      src[i] = ((src[i] << 16) | (src[i] >> 16)) ^ src[7 - i];

    }

  }



  result = svq1_decode_frame_header (&s->gb, s);



  if (result != 0)

  {

#ifdef DEBUG_SVQ1

    printf("Error in svq1_decode_frame_header %i\n",result);

#endif

    return result;

  }

  

  //FIXME this avoids some confusion for "B frames" without 2 references

  //this should be removed after libavcodec can handle more flaxible picture types & ordering

  if(s->pict_type==B_TYPE && s->last_picture.data[0]==NULL) return buf_size;

  

  if(avctx->hurry_up && s->pict_type==B_TYPE) return buf_size;



  if(MPV_frame_start(s, avctx) < 0)

      return -1;



  /* decode y, u and v components */

  for (i=0; i < 3; i++) {

    int linesize;

    if (i == 0) {

      width  = (s->width+15)&~15;

      height = (s->height+15)&~15;

      linesize= s->linesize;

    } else {

      if(s->flags&CODEC_FLAG_GRAY) break;

      width  = (s->width/4+15)&~15;

      height = (s->height/4+15)&~15;

      linesize= s->uvlinesize;

    }



    current  = s->current_picture.data[i];



    if(s->pict_type==B_TYPE){

        previous = s->next_picture.data[i];

    }else{

        previous = s->last_picture.data[i];

    }



    if (s->pict_type == I_TYPE) {

      /* keyframe */

      for (y=0; y < height; y+=16) {

	for (x=0; x < width; x+=16) {

	  result = svq1_decode_block_intra (&s->gb, &current[x], linesize);

	  if (result != 0)

	  {

#ifdef DEBUG_SVQ1

	    printf("Error in svq1_decode_block %i (keyframe)\n",result);

#endif

	    return result;

	  }

	}

	current += 16*linesize;

      }

    } else {

      svq1_pmv_t pmv[width/8+3];

      /* delta frame */

      memset (pmv, 0, ((width / 8) + 3) * sizeof(svq1_pmv_t));



      for (y=0; y < height; y+=16) {

	for (x=0; x < width; x+=16) {

	  result = svq1_decode_delta_block (s, &s->gb, &current[x], previous,

					    linesize, pmv, x, y);

	  if (result != 0)

	  {

#ifdef DEBUG_SVQ1

    printf("Error in svq1_decode_delta_block %i\n",result);

#endif

	    return result;

	  }

	}



	pmv[0].x =

	pmv[0].y = 0;



	current += 16*linesize;

      }

    }

  }

  

  *pict = *(AVFrame*)&s->current_picture;





  MPV_frame_end(s);

  

  *data_size=sizeof(AVFrame);

  return buf_size;

}
