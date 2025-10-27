static void clone_slice(H264Context *dst, H264Context *src)

{

    memcpy(dst->block_offset, src->block_offset, sizeof(dst->block_offset));

    dst->s.current_picture_ptr = src->s.current_picture_ptr;

    dst->s.current_picture     = src->s.current_picture;

    dst->s.linesize            = src->s.linesize;

    dst->s.uvlinesize          = src->s.uvlinesize;

    dst->s.first_field         = src->s.first_field;



    dst->prev_poc_msb          = src->prev_poc_msb;

    dst->prev_poc_lsb          = src->prev_poc_lsb;

    dst->prev_frame_num_offset = src->prev_frame_num_offset;

    dst->prev_frame_num        = src->prev_frame_num;

    dst->short_ref_count       = src->short_ref_count;



    memcpy(dst->short_ref,        src->short_ref,        sizeof(dst->short_ref));

    memcpy(dst->long_ref,         src->long_ref,         sizeof(dst->long_ref));

    memcpy(dst->default_ref_list, src->default_ref_list, sizeof(dst->default_ref_list));

    memcpy(dst->ref_list,         src->ref_list,         sizeof(dst->ref_list));



    memcpy(dst->dequant4_coeff,   src->dequant4_coeff,   sizeof(src->dequant4_coeff));

    memcpy(dst->dequant8_coeff,   src->dequant8_coeff,   sizeof(src->dequant8_coeff));

}
