static always_inline void mpeg_motion(MpegEncContext *s,

                               uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,

                               int field_based, int bottom_field, int field_select,

                               uint8_t **ref_picture, op_pixels_func (*pix_op)[4],

                               int motion_x, int motion_y, int h)

{

    uint8_t *ptr_y, *ptr_cb, *ptr_cr;

    int dxy, uvdxy, mx, my, src_x, src_y, uvsrc_x, uvsrc_y, v_edge_pos, uvlinesize, linesize;

    

#if 0    

if(s->quarter_sample)

{

    motion_x>>=1;

    motion_y>>=1;


#endif



    v_edge_pos = s->v_edge_pos >> field_based;

    linesize   = s->current_picture.linesize[0] << field_based;

    uvlinesize = s->current_picture.linesize[1] << field_based;



    dxy = ((motion_y & 1) << 1) | (motion_x & 1);

    src_x = s->mb_x* 16               + (motion_x >> 1);

    src_y =(s->mb_y<<(4-field_based)) + (motion_y >> 1);



    if (s->out_format == FMT_H263) {

        if((s->workaround_bugs & FF_BUG_HPEL_CHROMA) && field_based){

            mx = (motion_x>>1)|(motion_x&1);

            my = motion_y >>1;

            uvdxy = ((my & 1) << 1) | (mx & 1);

            uvsrc_x = s->mb_x* 8               + (mx >> 1);

            uvsrc_y = (s->mb_y<<(3-field_based)) + (my >> 1);

        }else{

            uvdxy = dxy | (motion_y & 2) | ((motion_x & 2) >> 1);

            uvsrc_x = src_x>>1;

            uvsrc_y = src_y>>1;


    }else if(s->out_format == FMT_H261){//even chroma mv's are full pel in H261

        mx = motion_x / 4;

        my = motion_y / 4;

        uvdxy = 0;

        uvsrc_x = s->mb_x*8 + mx;

        uvsrc_y = s->mb_y*8 + my;

    } else {

        if(s->chroma_y_shift){

            mx = motion_x / 2;

            my = motion_y / 2;

            uvdxy = ((my & 1) << 1) | (mx & 1);

            uvsrc_x = s->mb_x* 8               + (mx >> 1);

            uvsrc_y = (s->mb_y<<(3-field_based)) + (my >> 1);

        } else {

            if(s->chroma_x_shift){

            //Chroma422

                mx = motion_x / 2;

                uvdxy = ((motion_y & 1) << 1) | (mx & 1);

                uvsrc_x = s->mb_x* 8           + (mx >> 1);

                uvsrc_y = src_y;

            } else {

            //Chroma444

                uvdxy = dxy;

                uvsrc_x = src_x;

                uvsrc_y = src_y;






    ptr_y  = ref_picture[0] + src_y * linesize + src_x;

    ptr_cb = ref_picture[1] + uvsrc_y * uvlinesize + uvsrc_x;

    ptr_cr = ref_picture[2] + uvsrc_y * uvlinesize + uvsrc_x;



    if(   (unsigned)src_x > s->h_edge_pos - (motion_x&1) - 16

       || (unsigned)src_y >    v_edge_pos - (motion_y&1) - h){

            if(s->codec_id == CODEC_ID_MPEG2VIDEO ||

               s->codec_id == CODEC_ID_MPEG1VIDEO){

                av_log(s->avctx,AV_LOG_DEBUG,"MPEG motion vector out of boundary\n");

                return ;


            ff_emulated_edge_mc(s->edge_emu_buffer, ptr_y, s->linesize, 17, 17+field_based,

                             src_x, src_y<<field_based, s->h_edge_pos, s->v_edge_pos);

            ptr_y = s->edge_emu_buffer;

            if(!(s->flags&CODEC_FLAG_GRAY)){

                uint8_t *uvbuf= s->edge_emu_buffer+18*s->linesize;

                ff_emulated_edge_mc(uvbuf  , ptr_cb, s->uvlinesize, 9, 9+field_based, 

                                 uvsrc_x, uvsrc_y<<field_based, s->h_edge_pos>>1, s->v_edge_pos>>1);

                ff_emulated_edge_mc(uvbuf+16, ptr_cr, s->uvlinesize, 9, 9+field_based, 

                                 uvsrc_x, uvsrc_y<<field_based, s->h_edge_pos>>1, s->v_edge_pos>>1);

                ptr_cb= uvbuf;

                ptr_cr= uvbuf+16;





    if(bottom_field){ //FIXME use this for field pix too instead of the obnoxious hack which changes picture.data

        dest_y += s->linesize;

        dest_cb+= s->uvlinesize;

        dest_cr+= s->uvlinesize;




    if(field_select){

        ptr_y += s->linesize;

        ptr_cb+= s->uvlinesize;

        ptr_cr+= s->uvlinesize;




    pix_op[0][dxy](dest_y, ptr_y, linesize, h);

    

    if(!(s->flags&CODEC_FLAG_GRAY)){

        pix_op[s->chroma_x_shift][uvdxy](dest_cb, ptr_cb, uvlinesize, h >> s->chroma_y_shift);

        pix_op[s->chroma_x_shift][uvdxy](dest_cr, ptr_cr, uvlinesize, h >> s->chroma_y_shift);




