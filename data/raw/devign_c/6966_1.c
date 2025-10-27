int mpeg4_decode_picture_header(MpegEncContext * s)

{

    int time_incr, startcode, state, v;



 redo:

    /* search next start code */

    align_get_bits(&s->gb);

    state = 0xff;

    for(;;) {

        v = get_bits(&s->gb, 8);

        if (state == 0x000001) {

            state = ((state << 8) | v) & 0xffffff;

            startcode = state;

            break;

        }

        state = ((state << 8) | v) & 0xffffff;

        if( get_bits_count(&s->gb) > s->gb.size*8-32){

            printf("no VOP startcode found\n");

            return -1;

        }

    }

//printf("startcode %X %d\n", startcode, get_bits_count(&s->gb));

    if (startcode == 0x120) { // Video Object Layer

        int width, height, vo_ver_id;



        /* vol header */

        skip_bits(&s->gb, 1); /* random access */

        skip_bits(&s->gb, 8); /* vo_type */

        if (get_bits1(&s->gb) != 0) { /* is_ol_id */

            vo_ver_id = get_bits(&s->gb, 4); /* vo_ver_id */

            skip_bits(&s->gb, 3); /* vo_priority */

        } else {

            vo_ver_id = 1;

        }

        

        s->aspect_ratio_info= get_bits(&s->gb, 4);

	if(s->aspect_ratio_info == EXTENDET_PAR){

            skip_bits(&s->gb, 8); //par_width

            skip_bits(&s->gb, 8); // par_height

        }

        if(get_bits1(&s->gb)){ /* vol control parameter */

            printf("vol control parameter not supported\n");

            return -1;   

        }

        s->shape = get_bits(&s->gb, 2); /* vol shape */

        if(s->shape != RECT_SHAPE) printf("only rectangular vol supported\n");

        if(s->shape == GRAY_SHAPE && vo_ver_id != 1){

            printf("Gray shape not supported\n");

            skip_bits(&s->gb, 4);  //video_object_layer_shape_extension

        }



        skip_bits1(&s->gb);   /* marker */

        

        s->time_increment_resolution = get_bits(&s->gb, 16);

        s->time_increment_bits = av_log2(s->time_increment_resolution - 1) + 1;

        if (s->time_increment_bits < 1)

            s->time_increment_bits = 1;

        skip_bits1(&s->gb);   /* marker */



        if (get_bits1(&s->gb) != 0) {   /* fixed_vop_rate  */

            skip_bits(&s->gb, s->time_increment_bits);

        }



        if (s->shape != BIN_ONLY_SHAPE) {

            if (s->shape == RECT_SHAPE) {

                skip_bits1(&s->gb);   /* marker */

                width = get_bits(&s->gb, 13);

                skip_bits1(&s->gb);   /* marker */

                height = get_bits(&s->gb, 13);

                skip_bits1(&s->gb);   /* marker */

                if(width && height){ /* they should be non zero but who knows ... */

                    s->width = width;

                    s->height = height;

//                    printf("%d %d\n", width, height);

                }

            }

            

            if(get_bits1(&s->gb)) printf("interlaced not supported\n");   /* interlaced */

            if(!get_bits1(&s->gb)) printf("OBMC not supported\n");   /* OBMC Disable */

            if (vo_ver_id == 1) {

                s->vol_sprite_usage = get_bits1(&s->gb); /* vol_sprite_usage */

            } else {

                s->vol_sprite_usage = get_bits(&s->gb, 2); /* vol_sprite_usage */

            }

            if(s->vol_sprite_usage==STATIC_SPRITE) printf("Static Sprites not supported\n");

            if(s->vol_sprite_usage==STATIC_SPRITE || s->vol_sprite_usage==GMC_SPRITE){

                if(s->vol_sprite_usage==STATIC_SPRITE){

                    s->sprite_width = get_bits(&s->gb, 13);

                    skip_bits1(&s->gb); /* marker */

                    s->sprite_height= get_bits(&s->gb, 13);

                    skip_bits1(&s->gb); /* marker */

                    s->sprite_left  = get_bits(&s->gb, 13);

                    skip_bits1(&s->gb); /* marker */

                    s->sprite_top   = get_bits(&s->gb, 13);

                    skip_bits1(&s->gb); /* marker */

                }

                s->num_sprite_warping_points= get_bits(&s->gb, 6);

                s->sprite_warping_accuracy = get_bits(&s->gb, 2);

                s->sprite_brightness_change= get_bits1(&s->gb);

                if(s->vol_sprite_usage==STATIC_SPRITE)

                    s->low_latency_sprite= get_bits1(&s->gb);            

            }

            // FIXME sadct disable bit if verid!=1 && shape not rect

            

            if (get_bits1(&s->gb) == 1) {   /* not_8_bit */

                s->quant_precision = get_bits(&s->gb, 4); /* quant_precision */

                if(get_bits(&s->gb, 4)!=8) printf("N-bit not supported\n"); /* bits_per_pixel */

            } else {

                s->quant_precision = 5;

            }

            

            // FIXME a bunch of grayscale shape things

            if(get_bits1(&s->gb)) printf("Quant-Type not supported\n");  /* vol_quant_type */ //FIXME

            if(vo_ver_id != 1)

                 s->quarter_sample= get_bits1(&s->gb);

            else s->quarter_sample=0;



            if(!get_bits1(&s->gb)) printf("Complexity estimation not supported\n");

#if 0

            if(get_bits1(&s->gb)) printf("resync disable\n");

#else

            skip_bits1(&s->gb);   /* resync_marker_disabled */

#endif

            s->data_partioning= get_bits1(&s->gb);

            if(s->data_partioning){

                printf("data partitioning not supported\n");

                skip_bits1(&s->gb); // reversible vlc

            }

            

            if(vo_ver_id != 1) {

                s->new_pred= get_bits1(&s->gb);

                if(s->new_pred){

                    printf("new pred not supported\n");

                    skip_bits(&s->gb, 2); /* requested upstream message type */

                    skip_bits1(&s->gb); /* newpred segment type */

                }

                s->reduced_res_vop= get_bits1(&s->gb);

                if(s->reduced_res_vop) printf("reduced resolution VOP not supported\n");

            }

            else{

                s->new_pred=0;

                s->reduced_res_vop= 0;

            }



            s->scalability= get_bits1(&s->gb);

            if (s->scalability) {

                printf("bad scalability!!!\n");

                return -1;

            }

        }

//printf("end Data %X %d\n", show_bits(&s->gb, 32), get_bits_count(&s->gb)&0x7);

        goto redo;

    } else if (startcode == 0x1b2) { //userdata

        char buf[256];

        int i;

        int e;

        int ver, build;



//printf("user Data %X\n", show_bits(&s->gb, 32));

        buf[0]= show_bits(&s->gb, 8);

        for(i=1; i<256; i++){

            buf[i]= show_bits(&s->gb, 16)&0xFF;

            if(buf[i]==0) break;

            skip_bits(&s->gb, 8);

        }

        buf[255]=0;

        e=sscanf(buf, "DivX%dBuild%d", &ver, &build);

        if(e==2){

            s->divx_version= ver;

            s->divx_build= build;

            if(s->picture_number==0){

                printf("This file was encoded with DivX%d Build%d\n", ver, build);

                if(ver==500 && build==413){ //most likely all version are indeed totally buggy but i dunno for sure ...

                    printf("WARNING: this version of DivX is not MPEG4 compatible, trying to workaround these bugs...\n");

                }else{

                    printf("hmm, i havnt seen that version of divx yet, lets assume they fixed these bugs ...\n"

                           "using mpeg4 decoder, if it fails contact the developers (of ffmpeg)\n");

                }

            }

        }

//printf("User Data: %s\n", buf);

        goto redo;

    } else if (startcode != 0x1b6) { //VOP

        goto redo;

    }



    s->pict_type = get_bits(&s->gb, 2) + 1;	/* pict type: I = 0 , P = 1 */

//printf("pic: %d\n", s->pict_type); 

    time_incr=0;

    while (get_bits1(&s->gb) != 0) 

        time_incr++;



    check_marker(&s->gb, "before time_increment");

    s->time_increment= get_bits(&s->gb, s->time_increment_bits);

    if(s->pict_type!=B_TYPE){

        s->time_base+= time_incr;

        s->last_non_b_time[1]= s->last_non_b_time[0];

        s->last_non_b_time[0]= s->time_base*s->time_increment_resolution + s->time_increment;

    }else{

        s->time= (s->last_non_b_time[1]/s->time_increment_resolution + time_incr)*s->time_increment_resolution;

        s->time+= s->time_increment;

    }



    if(check_marker(&s->gb, "before vop_coded")==0 && s->picture_number==0){

        printf("hmm, seems the headers arnt complete, trying to guess time_increment_bits\n");

        for(s->time_increment_bits++ ;s->time_increment_bits<16; s->time_increment_bits++){

            if(get_bits1(&s->gb)) break;

        }

        printf("my guess is %d bits ;)\n",s->time_increment_bits);

    }

    /* vop coded */

    if (get_bits1(&s->gb) != 1)

        goto redo;

//printf("time %d %d %d || %d %d %d\n", s->time_increment_bits, s->time_increment, s->time_base,

//s->time, s->last_non_b_time[0], s->last_non_b_time[1]);  

    if (s->shape != BIN_ONLY_SHAPE && ( s->pict_type == P_TYPE

                          || (s->pict_type == S_TYPE && s->vol_sprite_usage==GMC_SPRITE))) {

        /* rounding type for motion estimation */

	s->no_rounding = get_bits1(&s->gb);

    } else {

	s->no_rounding = 0;

    }

//FIXME reduced res stuff



     if (s->shape != RECT_SHAPE) {

         if (s->vol_sprite_usage != 1 || s->pict_type != I_TYPE) {

             int width, height, hor_spat_ref, ver_spat_ref;

 

             width = get_bits(&s->gb, 13);

             skip_bits1(&s->gb);   /* marker */

             height = get_bits(&s->gb, 13);

             skip_bits1(&s->gb);   /* marker */

             hor_spat_ref = get_bits(&s->gb, 13); /* hor_spat_ref */

             skip_bits1(&s->gb);   /* marker */

             ver_spat_ref = get_bits(&s->gb, 13); /* ver_spat_ref */

         }

         skip_bits1(&s->gb); /* change_CR_disable */

 

         if (get_bits1(&s->gb) != 0) {

             skip_bits(&s->gb, 8); /* constant_alpha_value */

         }

     }

//FIXME complexity estimation stuff

     

     if (s->shape != BIN_ONLY_SHAPE) {

         skip_bits(&s->gb, 3); /* intra dc VLC threshold */

         //FIXME interlaced specific bits

     }



     if(s->pict_type == S_TYPE && (s->vol_sprite_usage==STATIC_SPRITE || s->vol_sprite_usage==GMC_SPRITE)){

         if(s->num_sprite_warping_points){

             mpeg4_decode_sprite_trajectory(s);

         }

         if(s->sprite_brightness_change) printf("sprite_brightness_change not supported\n");

         if(s->vol_sprite_usage==STATIC_SPRITE) printf("static sprite not supported\n");

     }



     if (s->shape != BIN_ONLY_SHAPE) {

         /* note: we do not use quant_precision to avoid problem if no

            MPEG4 vol header as it is found on some old opendivx

            movies */

         s->qscale = get_bits(&s->gb, 5);

         if(s->qscale==0){

             printf("Error, header damaged or not MPEG4 header (qscale=0)\n");

             return -1; // makes no sense to continue, as there is nothing left from the image then

         }

  

         if (s->pict_type != I_TYPE) {

             s->f_code = get_bits(&s->gb, 3);	/* fcode_for */

             if(s->f_code==0){

                 printf("Error, header damaged or not MPEG4 header (f_code=0)\n");

                 return -1; // makes no sense to continue, as the MV decoding will break very quickly

             }

         }

         if (s->pict_type == B_TYPE) {

             s->b_code = get_bits(&s->gb, 3);

//printf("b-code %d\n", s->b_code);

         }

//printf("quant:%d fcode:%d\n", s->qscale, s->f_code);



         if(!s->scalability){

             if (s->shape!=RECT_SHAPE && s->pict_type!=I_TYPE) {

                 skip_bits1(&s->gb); // vop shape coding type

             }

         }

     }

     s->picture_number++; // better than pic number==0 allways ;)

     return 0;

}
