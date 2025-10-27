static void init_block_index(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    ff_init_block_index(s);

    if (v->field_mode && !(v->second_field ^ v->tff)) {

        s->dest[0] += s->current_picture_ptr->f.linesize[0];

        s->dest[1] += s->current_picture_ptr->f.linesize[1];

        s->dest[2] += s->current_picture_ptr->f.linesize[2];

    }

}
