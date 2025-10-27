static void mov_text_style_cb(void *priv, const char style, int close)

{

    MovTextContext *s = priv;

    if (!close) {

        if (!(s->box_flags & STYL_BOX)) {   //first style entry



            s->style_attributes_temp = av_malloc(sizeof(*s->style_attributes_temp));



            if (!s->style_attributes_temp) {

                av_bprint_clear(&s->buffer);

                s->box_flags &= ~STYL_BOX;


            }



            s->style_attributes_temp->style_flag = 0;

            s->style_attributes_temp->style_start = AV_RB16(&s->text_pos);

        } else {

            if (s->style_attributes_temp->style_flag) { //break the style record here and start a new one

                s->style_attributes_temp->style_end = AV_RB16(&s->text_pos);

                av_dynarray_add(&s->style_attributes, &s->count, s->style_attributes_temp);

                s->style_attributes_temp = av_malloc(sizeof(*s->style_attributes_temp));

                if (!s->style_attributes_temp) {

                    mov_text_cleanup(s);

                    av_bprint_clear(&s->buffer);

                    s->box_flags &= ~STYL_BOX;


                }



                s->style_attributes_temp->style_flag = s->style_attributes[s->count - 1]->style_flag;

                s->style_attributes_temp->style_start = AV_RB16(&s->text_pos);

            } else {

                s->style_attributes_temp->style_flag = 0;

                s->style_attributes_temp->style_start = AV_RB16(&s->text_pos);

            }

        }

        switch (style){

        case 'b':

            s->style_attributes_temp->style_flag |= STYLE_FLAG_BOLD;

            break;

        case 'i':

            s->style_attributes_temp->style_flag |= STYLE_FLAG_ITALIC;

            break;

        case 'u':

            s->style_attributes_temp->style_flag |= STYLE_FLAG_UNDERLINE;

            break;

        }




    } else {

        s->style_attributes_temp->style_end = AV_RB16(&s->text_pos);

        av_dynarray_add(&s->style_attributes, &s->count, s->style_attributes_temp);



        s->style_attributes_temp = av_malloc(sizeof(*s->style_attributes_temp));



        if (!s->style_attributes_temp) {

            mov_text_cleanup(s);

            av_bprint_clear(&s->buffer);

            s->box_flags &= ~STYL_BOX;


        }



        s->style_attributes_temp->style_flag = s->style_attributes[s->count - 1]->style_flag;

        switch (style){

        case 'b':

            s->style_attributes_temp->style_flag &= ~STYLE_FLAG_BOLD;

            break;

        case 'i':

            s->style_attributes_temp->style_flag &= ~STYLE_FLAG_ITALIC;

            break;

        case 'u':

            s->style_attributes_temp->style_flag &= ~STYLE_FLAG_UNDERLINE;

            break;

        }

        if (s->style_attributes_temp->style_flag) { //start of new style record

            s->style_attributes_temp->style_start = AV_RB16(&s->text_pos);

        }

    }

    s->box_flags |= STYL_BOX;

}