static av_cold void iv_alloc_frames(Indeo3DecodeContext *s)

{

  int luma_width, luma_height, luma_pixels, chroma_width, chroma_height,

      chroma_pixels, i;

  unsigned int bufsize;



  luma_width   = (s->width  + 3) & (~3);

  luma_height  = (s->height + 3) & (~3);



  s->iv_frame[0].y_w = s->iv_frame[0].y_h =

    s->iv_frame[0].the_buf_size = 0;

  s->iv_frame[1].y_w = s->iv_frame[1].y_h =

    s->iv_frame[1].the_buf_size = 0;

  s->iv_frame[1].the_buf = NULL;



  chroma_width  = ((luma_width >> 2) + 3) & (~3);

  chroma_height = ((luma_height>> 2) + 3) & (~3);

  luma_pixels = luma_width * luma_height;

  chroma_pixels = chroma_width * chroma_height;



  bufsize = luma_pixels * 2 + luma_width * 3 +

    (chroma_pixels + chroma_width) * 4;



  if(!(s->iv_frame[0].the_buf = av_malloc(bufsize)))

    return;

  s->iv_frame[0].y_w = s->iv_frame[1].y_w = luma_width;

  s->iv_frame[0].y_h = s->iv_frame[1].y_h = luma_height;

  s->iv_frame[0].uv_w = s->iv_frame[1].uv_w = chroma_width;

  s->iv_frame[0].uv_h = s->iv_frame[1].uv_h = chroma_height;

  s->iv_frame[0].the_buf_size = bufsize;



  s->iv_frame[0].Ybuf = s->iv_frame[0].the_buf + luma_width;

  i = luma_pixels + luma_width * 2;

  s->iv_frame[1].Ybuf = s->iv_frame[0].the_buf + i;

  i += (luma_pixels + luma_width);

  s->iv_frame[0].Ubuf = s->iv_frame[0].the_buf + i;

  i += (chroma_pixels + chroma_width);

  s->iv_frame[1].Ubuf = s->iv_frame[0].the_buf + i;

  i += (chroma_pixels + chroma_width);

  s->iv_frame[0].Vbuf = s->iv_frame[0].the_buf + i;

  i += (chroma_pixels + chroma_width);

  s->iv_frame[1].Vbuf = s->iv_frame[0].the_buf + i;



  for(i = 1; i <= luma_width; i++)

    s->iv_frame[0].Ybuf[-i] = s->iv_frame[1].Ybuf[-i] =

      s->iv_frame[0].Ubuf[-i] = 0x80;



  for(i = 1; i <= chroma_width; i++) {

    s->iv_frame[1].Ubuf[-i] = 0x80;

    s->iv_frame[0].Vbuf[-i] = 0x80;

    s->iv_frame[1].Vbuf[-i] = 0x80;

    s->iv_frame[1].Vbuf[chroma_pixels+i-1] = 0x80;

  }

}
