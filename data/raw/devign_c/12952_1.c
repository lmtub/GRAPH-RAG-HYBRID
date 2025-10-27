static void common_end(FFV1Context *s){

    int i;



    for(i=0; i<s->plane_count; i++){

        PlaneContext *p= &s->plane[i];



        av_freep(&p->state);

    }

}
