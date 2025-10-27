void ff_mpeg_set_erpic(ERPicture *dst, Picture *src)

{

    int i;




    if (!src)

        return;



    dst->f = &src->f;

    dst->tf = &src->tf;



    for (i = 0; i < 2; i++) {

        dst->motion_val[i] = src->motion_val[i];

        dst->ref_index[i] = src->ref_index[i];

    }



    dst->mb_type = src->mb_type;

    dst->field_picture = src->field_picture;

}