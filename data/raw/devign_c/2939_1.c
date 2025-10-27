int msmpeg4_decode_picture_header(MpegEncContext * s)

{

    int code;



#if 0

{

int i;

for(i=0; i<s->gb.size_in_bits; i++)

    printf("%d", get_bits1(&s->gb));

//    get_bits1(&s->gb);

printf("END\n");



#endif



    if(s->msmpeg4_version==1){

        int start_code, num;

        start_code = (get_bits(&s->gb, 16)<<16) | get_bits(&s->gb, 16);

        if(start_code!=0x00000100){

            fprintf(stderr, "invalid startcode\n");





        num= get_bits(&s->gb, 5); // frame number */




    s->pict_type = get_bits(&s->gb, 2) + 1;

    if (s->pict_type != I_TYPE &&

        s->pict_type != P_TYPE){

        fprintf(stderr, "invalid picture type\n");



#if 0

{

    static int had_i=0;

    if(s->pict_type == I_TYPE) had_i=1;

    if(!had_i) return -1;


#endif

    s->qscale = get_bits(&s->gb, 5);







    if (s->pict_type == I_TYPE) {

        code = get_bits(&s->gb, 5); 

        if(s->msmpeg4_version==1){

            if(code==0 || code>s->mb_height){

                fprintf(stderr, "invalid slice height %d\n", code);





            s->slice_height = code;

        }else{

            /* 0x17: one slice, 0x18: two slices, ... */

            if (code < 0x17){

                fprintf(stderr, "error, slice code was %X\n", code);





            s->slice_height = s->mb_height / (code - 0x16);




        switch(s->msmpeg4_version){

        case 1:

        case 2:

            s->rl_chroma_table_index = 2;

            s->rl_table_index = 2;



            s->dc_table_index = 0; //not used

            break;

        case 3:

            s->rl_chroma_table_index = decode012(&s->gb);

            s->rl_table_index = decode012(&s->gb);



            s->dc_table_index = get_bits1(&s->gb);

            break;

        case 4:

            msmpeg4_decode_ext_header(s, (2+5+5+17+7)/8);



            if(s->bit_rate > MBAC_BITRATE) s->per_mb_rl_table= get_bits1(&s->gb);

            else                           s->per_mb_rl_table= 0;

            

            if(!s->per_mb_rl_table){

                s->rl_chroma_table_index = decode012(&s->gb);

                s->rl_table_index = decode012(&s->gb);




            s->dc_table_index = get_bits1(&s->gb);

            s->inter_intra_pred= 0;

            break;


        s->no_rounding = 1;

/*	printf("qscale:%d rlc:%d rl:%d dc:%d mbrl:%d slice:%d   \n", 

		s->qscale,

		s->rl_chroma_table_index,

		s->rl_table_index, 

		s->dc_table_index,

                s->per_mb_rl_table,

                s->slice_height);*/

    } else {

        switch(s->msmpeg4_version){

        case 1:

        case 2:

            if(s->msmpeg4_version==1)

                s->use_skip_mb_code = 1;

            else

                s->use_skip_mb_code = get_bits1(&s->gb);

            s->rl_table_index = 2;

            s->rl_chroma_table_index = s->rl_table_index;

            s->dc_table_index = 0; //not used

            s->mv_table_index = 0;

            break;

        case 3:

            s->use_skip_mb_code = get_bits1(&s->gb);

            s->rl_table_index = decode012(&s->gb);

            s->rl_chroma_table_index = s->rl_table_index;



            s->dc_table_index = get_bits1(&s->gb);



            s->mv_table_index = get_bits1(&s->gb);

            break;

        case 4:

            s->use_skip_mb_code = get_bits1(&s->gb);



            if(s->bit_rate > MBAC_BITRATE) s->per_mb_rl_table= get_bits1(&s->gb);

            else                           s->per_mb_rl_table= 0;



            if(!s->per_mb_rl_table){

                s->rl_table_index = decode012(&s->gb);

                s->rl_chroma_table_index = s->rl_table_index;




            s->dc_table_index = get_bits1(&s->gb);



            s->mv_table_index = get_bits1(&s->gb);

            s->inter_intra_pred= (s->width*s->height < 320*240 && s->bit_rate<=II_BITRATE);

            break;


/*	printf("skip:%d rl:%d rlc:%d dc:%d mv:%d mbrl:%d qp:%d   \n", 

		s->use_skip_mb_code, 

		s->rl_table_index, 

		s->rl_chroma_table_index, 

		s->dc_table_index,

		s->mv_table_index,

                s->per_mb_rl_table,

                s->qscale);*/

	if(s->flipflop_rounding){

	    s->no_rounding ^= 1;

	}else{

	    s->no_rounding = 0;



//printf("%d %d %d %d %d\n", s->pict_type, s->bit_rate, s->inter_intra_pred, s->width, s->height);



    s->esc3_level_length= 0;

    s->esc3_run_length= 0;



#ifdef DEBUG

    printf("*****frame %d:\n", frame_count++);

#endif

    return 0;
